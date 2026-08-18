import argparse
import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple opt/SAR ReID datasets into one Merged-style dataset."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=(
            "Source dataset in NAME=ROOT format. Can be passed multiple times. "
            "Defaults to hoss and mos_ship no-leak roots."
        ),
    )
    parser.add_argument(
        "--dst_root",
        default="/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/HOSS-MOSShip-Merged",
        type=str,
        help="Output dataset root. Original datasets are never modified.",
    )
    parser.add_argument("--pid_start", default=0, type=int, help="First new PID in the merged dataset.")
    parser.add_argument(
        "--copy_mode",
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How to place images in the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove dst_root before writing. Refuses to overwrite any source root.",
    )
    parser.add_argument(
        "--eval_common_only",
        action="store_true",
        help=(
            "Keep only evaluation PIDs present in both query and gallery for each "
            "source. Training images are not affected."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Print planned merge without writing files.")
    return parser.parse_args()


def default_sources():
    return [
        (
            "hoss",
            "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/Merged",
        ),
        (
            "mos_ship",
            "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/MOS-Ship-ReID-SceneID-NoLeak",
        ),
    ]


def parse_sources(source_args):
    if not source_args:
        return default_sources()

    sources = []
    for item in source_args:
        if "=" not in item:
            raise ValueError(f"Invalid --source '{item}'. Expected NAME=ROOT.")
        name, root = item.split("=", 1)
        name = name.strip()
        root = root.strip()
        if not name or not root:
            raise ValueError(f"Invalid --source '{item}'. Expected NAME=ROOT.")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        sources.append((safe_name, root))
    return sources


def collect_images(split_dir):
    if not split_dir.exists():
        raise FileNotFoundError(str(split_dir))
    return sorted(
        path for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def extract_pid(path):
    stem = path.stem.lower()
    patterns = [
        r"^(-?\d+)_s\d+c\d+_(opt|sar)$",
        r"^(-?\d+)_s\d+c\d+$",
        r"^(-?\d+)[_-]",
        r"^(-?\d+)$",
        r"(?:pid|id)[_-]?(-?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot parse PID from filename: {path.name}")


def extract_modality(path):
    stem = path.stem.lower()
    if "_" in stem:
        suffix = stem.rsplit("_", 1)[-1]
        if suffix in {"opt", "rgb", "visible", "vis"}:
            return "opt"
        if suffix in {"sar", "ir"}:
            return "sar"

    parts = [part.lower() for part in path.parts]
    if any(part in {"opt", "rgb", "visible", "vis"} for part in parts):
        return "opt"
    if any(part in {"sar", "ir"} for part in parts):
        return "sar"
    raise ValueError(f"Cannot parse modality from path: {path}")


def safe_prepare_output(dst_root, source_roots, overwrite, dry_run):
    dst = Path(dst_root)
    dst_abs = dst.resolve()
    source_abs = {Path(root).resolve() for _, root in source_roots}

    if dst_abs in source_abs:
        raise ValueError("dst_root must not be the same as a source root.")

    if dry_run:
        return

    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{dst} already exists and is not empty. Use --overwrite to rebuild it."
            )
        shutil.rmtree(dst)

    for rel in SPLITS.values():
        (dst / rel).mkdir(parents=True, exist_ok=True)


def place_file(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def format_summary(stats, pid_map, intersections, args, sources):
    lines = []
    lines.append("=" * 80)
    lines.append("MERGED REID DATASET BUILD SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Output root: {args.dst_root}")
    lines.append(f"Copy mode:   {args.copy_mode}")
    lines.append(f"Dry run:     {args.dry_run}")
    lines.append(f"Eval common: {args.eval_common_only}")
    lines.append("")
    lines.append("[sources]")
    for name, root in sources:
        lines.append(f"  {name}: {root}")

    lines.append("")
    for split in ("train", "query", "gallery"):
        split_stats = stats["split"][split]
        lines.append(f"[{split}]")
        lines.append(f"  images:     {split_stats['images']}")
        lines.append(f"  ids:        {len(split_stats['pids'])}")
        lines.append(f"  modalities: {dict(split_stats['modalities'])}")
        lines.append(f"  sources:    {dict(split_stats['sources'])}")

    lines.append("")
    lines.append("[ID intersections]")
    for name, count in intersections.items():
        lines.append(f"  {name}: {count}")

    lines.append("")
    lines.append("[pid mapping]")
    lines.append(f"  merged ids: {len(pid_map)}")
    return "\n".join(lines)


def main():
    args = parse_args()
    sources = parse_sources(args.source)
    source_roots = [(name, Path(root)) for name, root in sources]
    dst_root = Path(args.dst_root)

    for name, root in source_roots:
        if not root.exists():
            raise FileNotFoundError(f"Source '{name}' does not exist: {root}")
        for rel in SPLITS.values():
            split_dir = root / rel
            if not split_dir.exists():
                raise FileNotFoundError(f"Source '{name}' missing required split: {split_dir}")

    safe_prepare_output(dst_root, source_roots, args.overwrite, args.dry_run)

    common_eval_pids = {}
    if args.eval_common_only:
        for source_name, source_root in source_roots:
            query_pids = {
                extract_pid(path)
                for path in collect_images(source_root / SPLITS["query"])
            }
            gallery_pids = {
                extract_pid(path)
                for path in collect_images(source_root / SPLITS["gallery"])
            }
            common_eval_pids[source_name] = query_pids & gallery_pids

    pid_map = {}
    next_pid = int(args.pid_start)
    next_img_id = 1
    manifest_rows = []
    stats = {"split": defaultdict(lambda: {"images": 0, "pids": set(), "modalities": Counter(), "sources": Counter()})}

    def map_pid(source_name, old_pid):
        nonlocal next_pid
        key = (source_name, int(old_pid))
        if key not in pid_map:
            pid_map[key] = next_pid
            next_pid += 1
        return pid_map[key]

    for source_name, source_root in source_roots:
        for split, rel in SPLITS.items():
            out_dir = dst_root / rel
            for src_path in collect_images(source_root / rel):
                old_pid = extract_pid(src_path)
                if old_pid == -1:
                    continue
                if (
                    args.eval_common_only
                    and split in ("query", "gallery")
                    and old_pid not in common_eval_pids[source_name]
                ):
                    continue
                modality = extract_modality(src_path)
                new_pid = map_pid(source_name, old_pid)
                ext = src_path.suffix.lower()
                out_name = f"{new_pid:04d}_s{next_img_id}c001_{modality}{ext}"
                out_path = out_dir / out_name

                if not args.dry_run:
                    place_file(src_path, out_path, args.copy_mode)

                manifest_rows.append(
                    {
                        "source_name": source_name,
                        "split": split,
                        "source_path": str(src_path),
                        "output_path": str(out_path),
                        "old_pid": old_pid,
                        "new_pid": new_pid,
                        "modality": modality,
                    }
                )
                split_stats = stats["split"][split]
                split_stats["images"] += 1
                split_stats["pids"].add(new_pid)
                split_stats["modalities"][modality] += 1
                split_stats["sources"][source_name] += 1
                next_img_id += 1

    train_ids = stats["split"]["train"]["pids"]
    query_ids = stats["split"]["query"]["pids"]
    gallery_ids = stats["split"]["gallery"]["pids"]
    intersections = {
        "query & gallery": len(query_ids & gallery_ids),
        "train & query": len(train_ids & query_ids),
        "train & gallery": len(train_ids & gallery_ids),
    }
    summary = format_summary(stats, pid_map, intersections, args, sources)
    print(summary)

    if args.dry_run:
        return 0

    with open(dst_root / "merge_manifest.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_name",
            "split",
            "source_path",
            "output_path",
            "old_pid",
            "new_pid",
            "modality",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(dst_root / "pid_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_name", "old_pid", "new_pid"])
        writer.writeheader()
        for (source_name, old_pid), new_pid in sorted(pid_map.items(), key=lambda item: item[1]):
            writer.writerow({"source_name": source_name, "old_pid": old_pid, "new_pid": new_pid})

    with open(dst_root / "merge_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\nWrote merged dataset to: {dst_root}")
    print(f"Wrote manifest: {dst_root / 'merge_manifest.csv'}")
    print(f"Wrote PID mapping: {dst_root / 'pid_mapping.csv'}")
    print(f"Wrote summary: {dst_root / 'merge_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
