import argparse
import json
import os
import sys

import torch

from config import cfg
from model.backbones.vit_transoss import vit_base_patch16_224_TransOSS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit whether a checkpoint is a usable TransOSS/SDF-Net pretrain weight."
    )
    parser.add_argument("--config_file", default="configs/SDF-Net_Multi_Paired.yml", type=str)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Checkpoint path. Defaults to MODEL.PRETRAIN_PATH from the config.",
    )
    parser.add_argument("--json", default=None, type=str, help="Optional path to save JSON report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 when required SAR branch keys are missing.",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping.")
    return checkpoint


def normalize_keys(state_dict, target_keys):
    normalized = {}
    original_for_normalized = {}
    for key, value in state_dict.items():
        clean = key.replace("module.", "")
        if clean not in target_keys and clean.startswith("base."):
            clean = clean[len("base.") :]
        normalized[clean] = value
        original_for_normalized[clean] = key
    return normalized, original_for_normalized


def build_backbone(local_cfg):
    return vit_base_patch16_224_TransOSS(
        img_size=local_cfg.INPUT.SIZE_TRAIN,
        mie_coe=local_cfg.MODEL.MIE_COE,
        camera=2 if local_cfg.MODEL.MIE else 0,
        stride_size=local_cfg.MODEL.STRIDE_SIZE,
        drop_path_rate=local_cfg.MODEL.DROP_PATH,
        drop_rate=local_cfg.MODEL.DROP_OUT,
        attn_drop_rate=local_cfg.MODEL.ATT_DROP_RATE,
        sse=local_cfg.MODEL.SSE,
        disentangle=local_cfg.MODEL.DISENTANGLE,
        struct_layer_index=local_cfg.MODEL.STRUCT_LAYER_INDEX,
    )


def tensor_diff_stats(a, b):
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose": bool(torch.allclose(a.float(), b.float(), atol=1e-7, rtol=1e-5)),
    }


def main():
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts or [])
    cfg.freeze()

    checkpoint_path = args.checkpoint or cfg.MODEL.PRETRAIN_PATH
    if not checkpoint_path:
        raise ValueError("No checkpoint path was provided and MODEL.PRETRAIN_PATH is empty.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    backbone = build_backbone(cfg)
    target_state = backbone.state_dict()
    target_keys = set(target_state.keys())
    raw_state = load_checkpoint(checkpoint_path)
    state, original_keys = normalize_keys(raw_state, target_keys)

    required_sar_keys = ["patch_embed_SAR.proj.weight", "patch_embed_SAR.proj.bias"]
    required_rgb_keys = ["patch_embed.proj.weight", "patch_embed.proj.bias"]

    missing_required_sar = [key for key in required_sar_keys if key not in state]
    missing_required_rgb = [key for key in required_rgb_keys if key not in state]
    missing_target = []
    shape_mismatch = []
    loaded = []
    for key, target_value in target_state.items():
        if key not in state:
            missing_target.append(key)
            continue
        if tuple(state[key].shape) != tuple(target_value.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(state[key].shape),
                    "target_shape": list(target_value.shape),
                }
            )
        else:
            loaded.append(key)

    unexpected = [key for key in state.keys() if key not in target_keys]

    sar_rgb_diff = None
    if "patch_embed.proj.weight" in state and "patch_embed_SAR.proj.weight" in state:
        if tuple(state["patch_embed.proj.weight"].shape) == tuple(
            state["patch_embed_SAR.proj.weight"].shape
        ):
            sar_rgb_diff = tensor_diff_stats(
                state["patch_embed.proj.weight"], state["patch_embed_SAR.proj.weight"]
            )

    report = {
        "checkpoint": checkpoint_path,
        "raw_key_count": len(raw_state),
        "normalized_key_count": len(state),
        "target_key_count": len(target_state),
        "loaded_shape_matched": len(loaded),
        "missing_required_rgb": missing_required_rgb,
        "missing_required_sar": missing_required_sar,
        "missing_target_count": len(missing_target),
        "missing_target_first20": missing_target[:20],
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch_first20": shape_mismatch[:20],
        "unexpected_count": len(unexpected),
        "unexpected_first20": unexpected[:20],
        "sar_rgb_patch_embed_diff": sar_rgb_diff,
        "sar_original_keys": {
            key: original_keys.get(key) for key in required_sar_keys if key in original_keys
        },
    }

    print("=" * 80)
    print("TRANSOSS PRETRAIN CHECKPOINT AUDIT")
    print("=" * 80)
    print(f"Checkpoint:             {checkpoint_path}")
    print(f"Raw keys:               {report['raw_key_count']}")
    print(f"Target backbone keys:   {report['target_key_count']}")
    print(f"Shape-matched keys:     {report['loaded_shape_matched']}")
    print(f"Missing RGB keys:       {missing_required_rgb}")
    print(f"Missing SAR keys:       {missing_required_sar}")
    print(f"Shape mismatches:       {len(shape_mismatch)}")
    print(f"Missing target keys:    {len(missing_target)}")
    if sar_rgb_diff is not None:
        print(
            "RGB/SAR patch diff:     "
            f"max={sar_rgb_diff['max_abs']:.6g}, "
            f"mean={sar_rgb_diff['mean_abs']:.6g}, "
            f"allclose={sar_rgb_diff['allclose']}"
        )

    if missing_required_sar:
        print("\nFAIL: checkpoint does not contain a pretrained SAR patch embedding.")
        print("      Do not use it as the final stage-1 TransOSS pretrain weight.")
    elif sar_rgb_diff and sar_rgb_diff["allclose"]:
        print("\nWARN: SAR and RGB patch embeddings are numerically identical.")
        print("      This can be only an initialization checkpoint, not a completed opt-SAR pretrain.")
    else:
        print("\nOK: required SAR branch keys are present.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to: {args.json}")

    if args.strict and missing_required_sar:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
