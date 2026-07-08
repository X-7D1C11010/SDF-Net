import argparse
import csv
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset

from config import cfg
from model.backbones.vit_transoss import vit_base_patch16_224_TransOSS


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
OPT_ALIASES = ("opt", "rgb", "visible", "vis")
SAR_ALIASES = ("sar", "ir")


def parse_args():
    parser = argparse.ArgumentParser(description="TransOSS opt-SAR contrastive pretraining")
    parser.add_argument("--config_file", default="configs/pretrain_transoss_contrastive.yml", type=str)
    parser.add_argument(
        "--data_root",
        default=None,
        type=str,
        help="Root with opt/ and sar/ subfolders. Defaults to DATASETS.ROOT_DIR.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        type=str,
        help="Optional CSV/TSV with opt_path and sar_path columns, or two path columns.",
    )
    parser.add_argument("--init_weight", default=None, type=str, help="Initial ImageNet/ViT weight.")
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--structure_weight", default=0.1, type=float)
    parser.add_argument(
        "--feature_mode",
        default="shared",
        choices=["shared", "fused"],
        help="Feature used for InfoNCE when DISENTANGLE is enabled.",
    )
    parser.add_argument("--max_pairs", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_img_wh(img):
    width, height = img.size
    scaled_w = width * 0.75
    scaled_h = height * 0.75
    return (
        (scaled_w / 93 - 0.434) / 0.031,
        (scaled_h / 427 - 0.461) / 0.031,
        scaled_h / scaled_w,
    )


def strip_modality_suffix(name):
    lower = name.lower()
    for alias in OPT_ALIASES + SAR_ALIASES:
        for sep in ("_", "-"):
            suffix = sep + alias
            if lower.endswith(suffix):
                return lower[: -len(suffix)]
    if "_" in lower:
        stem, tail = lower.rsplit("_", 1)
        if tail in OPT_ALIASES + SAR_ALIASES:
            return stem
    return lower


def collect_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(root, file_name))
    return sorted(images)


def read_manifest(path):
    delimiter = "\t" if path.lower().endswith(".tsv") else ","
    pairs = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                opt_path = row.get("opt_path") or row.get("opt") or row.get("rgb")
                sar_path = row.get("sar_path") or row.get("sar")
                if opt_path and sar_path:
                    pairs.append((opt_path, sar_path))
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0], row[1]))
    return pairs


class CrossModalPairDataset(Dataset):
    def __init__(self, data_root=None, manifest=None, transform=None, max_pairs=None):
        self.transform = transform
        if manifest:
            pairs = read_manifest(manifest)
        else:
            if not data_root:
                raise ValueError("Either data_root or manifest must be provided.")
            opt_dir = os.path.join(data_root, "opt")
            sar_dir = os.path.join(data_root, "sar")
            if not os.path.isdir(opt_dir) or not os.path.isdir(sar_dir):
                raise FileNotFoundError(
                    f"Expected opt/ and sar/ subfolders under {data_root}."
                )
            opt_map = {
                strip_modality_suffix(os.path.splitext(os.path.basename(path))[0]): path
                for path in collect_images(opt_dir)
            }
            sar_map = {
                strip_modality_suffix(os.path.splitext(os.path.basename(path))[0]): path
                for path in collect_images(sar_dir)
            }
            keys = sorted(set(opt_map.keys()) & set(sar_map.keys()))
            pairs = [(opt_map[key], sar_map[key]) for key in keys]

        if max_pairs is not None:
            pairs = pairs[: int(max_pairs)]
        if not pairs:
            raise RuntimeError("No opt-SAR pretraining pairs were found.")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, path):
        img = Image.open(path).convert("RGB")
        img_wh = torch.tensor(compute_img_wh(img), dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_wh

    def __getitem__(self, index):
        opt_path, sar_path = self.pairs[index]
        opt_img, opt_wh = self._read_image(opt_path)
        sar_img, sar_wh = self._read_image(sar_path)
        return opt_img, sar_img, opt_wh, sar_wh


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


def extract_pretrain_feature(outputs, disentangle, feature_mode):
    if disentangle:
        shared_feat, spec_feat, f_struct = outputs
        if feature_mode == "fused":
            return shared_feat + spec_feat, f_struct
        return shared_feat, f_struct
    feat, f_struct = outputs
    return feat, f_struct


def bidirectional_infonce(opt_feat, sar_feat, temperature):
    opt_feat = F.normalize(opt_feat, dim=1)
    sar_feat = F.normalize(sar_feat, dim=1)
    logits = torch.matmul(opt_feat, sar_feat.t()) / temperature
    labels = torch.arange(opt_feat.size(0), device=opt_feat.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def save_backbone(path, backbone, epoch, args, avg_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": backbone.state_dict(),
            "epoch": int(epoch),
            "avg_loss": float(avg_loss),
            "feature_mode": args.feature_mode,
            "temperature": float(args.temperature),
            "structure_weight": float(args.structure_weight),
        },
        path,
    )


def main():
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts or [])
    cfg.freeze()

    seed = args.seed if args.seed is not None else cfg.SOLVER.SEED
    set_seed(seed)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    logger = logging.getLogger("transoss.pretrain")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = args.data_root or cfg.DATASETS.ROOT_DIR
    output_dir = args.output_dir or cfg.OUTPUT_DIR
    epochs = args.epochs or cfg.SOLVER.MAX_EPOCHS
    batch_size = args.batch_size or cfg.SOLVER.IMS_PER_BATCH
    lr = args.lr or cfg.SOLVER.BASE_LR
    num_workers = args.num_workers if args.num_workers is not None else cfg.DATALOADER.NUM_WORKERS
    init_weight = args.init_weight if args.init_weight is not None else cfg.MODEL.PRETRAIN_PATH

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Seed: %s", seed)
    logger.info("Device: %s", device)
    logger.info("Data root: %s", data_root)
    logger.info("Manifest: %s", args.manifest)
    logger.info("Output dir: %s", output_dir)
    logger.info("Initial weight: %s", init_weight)

    transform = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    dataset = CrossModalPairDataset(
        data_root=data_root,
        manifest=args.manifest,
        transform=transform,
        max_pairs=args.max_pairs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    if len(loader) == 0:
        raise RuntimeError("No full pretraining batches. Reduce batch_size or add more pairs.")
    logger.info("Pretrain pairs: %d | batches/epoch: %d", len(dataset), len(loader))

    backbone = build_backbone(cfg).to(device)
    if init_weight:
        if not os.path.exists(init_weight):
            raise FileNotFoundError(init_weight)
        backbone.load_param(init_weight)

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = amp.GradScaler(enabled=(device == "cuda"))
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        backbone.train()
        start = time.time()
        running_loss = 0.0
        running_infonce = 0.0
        running_struct = 0.0

        for opt_img, sar_img, opt_wh, sar_wh in loader:
            opt_img = opt_img.to(device, non_blocking=True)
            sar_img = sar_img.to(device, non_blocking=True)
            opt_wh = opt_wh.to(device, non_blocking=True)
            sar_wh = sar_wh.to(device, non_blocking=True)

            imgs = torch.cat([opt_img, sar_img], dim=0)
            camids = torch.cat(
                [
                    torch.zeros(opt_img.size(0), dtype=torch.long),
                    torch.ones(sar_img.size(0), dtype=torch.long),
                ],
                dim=0,
            ).to(device, non_blocking=True)
            img_wh = torch.cat([opt_wh, sar_wh], dim=0)

            optimizer.zero_grad()
            with amp.autocast(enabled=(device == "cuda")):
                outputs = backbone(imgs, cam_label=camids, img_wh=img_wh)
                feat, f_struct = extract_pretrain_feature(
                    outputs, cfg.MODEL.DISENTANGLE, args.feature_mode
                )
                opt_feat = feat[: opt_img.size(0)]
                sar_feat = feat[opt_img.size(0) :]
                infonce_loss = bidirectional_infonce(opt_feat, sar_feat, args.temperature)
                if f_struct is not None and args.structure_weight > 0:
                    struct_loss = F.mse_loss(
                        f_struct[: opt_img.size(0)], f_struct[opt_img.size(0) :]
                    )
                else:
                    struct_loss = infonce_loss.new_tensor(0.0)
                loss = infonce_loss + args.structure_weight * struct_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())
            running_infonce += float(infonce_loss.detach().cpu())
            running_struct += float(struct_loss.detach().cpu())

        avg_loss = running_loss / len(loader)
        avg_infonce = running_infonce / len(loader)
        avg_struct = running_struct / len(loader)
        logger.info(
            "Epoch[%d/%d] loss=%.4f infonce=%.4f struct=%.4f time=%.1fs",
            epoch,
            epochs,
            avg_loss,
            avg_infonce,
            avg_struct,
            time.time() - start,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_backbone(
                os.path.join(output_dir, "transoss_contrastive_best.pth"),
                backbone,
                epoch,
                args,
                avg_loss,
            )

        if cfg.SOLVER.CHECKPOINT_PERIOD > 0 and epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            save_backbone(
                os.path.join(output_dir, f"transoss_contrastive_ep{epoch}.pth"),
                backbone,
                epoch,
                args,
                avg_loss,
            )

    save_backbone(
        os.path.join(output_dir, "transoss_contrastive_last.pth"),
        backbone,
        epochs,
        args,
        avg_loss,
    )
    logger.info("Pretraining finished. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
