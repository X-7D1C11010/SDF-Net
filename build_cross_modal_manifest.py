import argparse
import csv
import os


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MODALITY_SUFFIXES = ("opt", "rgb", "visible", "vis", "sar", "ir")


def parse_args():
    parser = argparse.ArgumentParser(description="Build an opt-SAR pair manifest for pretraining.")
    parser.add_argument("--opt_dir", required=True, type=str)
    parser.add_argument("--sar_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--key_mode",
        default="strip_modality_suffix",
        choices=["strip_modality_suffix", "stem"],
        help="How filenames are paired.",
    )
    return parser.parse_args()


def collect_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def strip_modality_suffix(stem):
    lower = stem.lower()
    for suffix in MODALITY_SUFFIXES:
        for sep in ("_", "-"):
            marker = sep + suffix
            if lower.endswith(marker):
                return lower[: -len(marker)]
    if "_" in lower:
        prefix, tail = lower.rsplit("_", 1)
        if tail in MODALITY_SUFFIXES:
            return prefix
    return lower


def make_key(path, mode):
    stem = os.path.splitext(os.path.basename(path))[0]
    if mode == "stem":
        return stem.lower()
    return strip_modality_suffix(stem)


def main():
    args = parse_args()
    opt_paths = collect_images(args.opt_dir)
    sar_paths = collect_images(args.sar_dir)
    opt_map = {make_key(path, args.key_mode): path for path in opt_paths}
    sar_map = {make_key(path, args.key_mode): path for path in sar_paths}
    keys = sorted(set(opt_map.keys()) & set(sar_map.keys()))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["opt_path", "sar_path", "pair_key"])
        writer.writeheader()
        for key in keys:
            writer.writerow(
                {
                    "opt_path": opt_map[key],
                    "sar_path": sar_map[key],
                    "pair_key": key,
                }
            )

    print("=" * 80)
    print("CROSS-MODAL MANIFEST")
    print("=" * 80)
    print(f"OPT images:     {len(opt_paths)}")
    print(f"SAR images:     {len(sar_paths)}")
    print(f"Paired samples: {len(keys)}")
    print(f"Output:         {args.output}")
    if len(keys) == 0:
        print("WARNING: no matching keys were found. Try --key_mode stem or prepare a custom manifest.")


if __name__ == "__main__":
    main()
