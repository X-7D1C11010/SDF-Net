import argparse
import csv
import os
from collections import Counter

import numpy as np

from config import cfg
from datasets.merged import MergedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Merged ReID dataset layout and ID distribution")
    parser.add_argument("--config_file", default="configs/SDF-Net_Test.yml", type=str)
    parser.add_argument("--root", default=None, type=str, help="override DATASETS.ROOT_DIR")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="audit query/gallery only; use this for independent test roots without training images",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using command-line KEY VALUE pairs",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def summarize_split(name, split, dataset):
    pids = [pid for _, pid, _, _ in split]
    cams = [camid for _, _, camid, _ in split]
    modalities = [dataset._extract_modality(path) for path, _, _, _ in split]
    per_pid = Counter(pids)

    print(f"\n[{name}]")
    print(f"  images:     {len(split)}")
    print(f"  ids:        {len(set(pids))}")
    print(f"  cameras:    {dict(Counter(cams))}")
    print(f"  modalities: {dict(Counter(modalities))}")
    if per_pid:
        counts = np.asarray(list(per_pid.values()))
        print(
            "  imgs/id:    "
            f"min={counts.min()}, p25={np.percentile(counts, 25):.1f}, "
            f"median={np.median(counts):.1f}, mean={counts.mean():.1f}, "
            f"p75={np.percentile(counts, 75):.1f}, max={counts.max()}"
        )
    return set(pids), per_pid


def summarize_pairs(dataset):
    paired_pids = set()
    for pair in dataset.train_pair:
        for _, pid, _, _ in pair:
            paired_pids.add(pid)
    print("\n[train pairs]")
    print(f"  pairs:      {len(dataset.train_pair)}")
    print(f"  paired ids: {len(paired_pids)}")
    if len(dataset.train_pair) > max(100000, dataset.num_train_imgs * 100):
        print(
            "  warning:    pair count is very large; this is usually caused by "
            "cartesian opt x sar pairing for IDs with many samples."
        )
        print(
            "              Consider DATASETS.PAIR_STRATEGY zip or "
            "DATASETS.MAX_PAIR_PER_ID 64 when using train_pair.py."
        )


def print_intersection(name, left, right):
    inter = left & right
    print(f"  {name}: {len(inter)}")
    return inter


def summarize_crop_manifest(root):
    manifest_path = os.path.join(root, "crop_manifest.csv")
    if not os.path.exists(manifest_path):
        return

    by_partition = {"train": set(), "query": set(), "gallery": set()}
    per_raw = Counter()
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"raw_identity", "partition"}.issubset(set(reader.fieldnames or [])):
            return
        for row in reader:
            raw_identity = row.get("raw_identity")
            partition = row.get("partition")
            if raw_identity and partition in by_partition:
                by_partition[partition].add(raw_identity)
                per_raw[raw_identity] += 1

    train_query = by_partition["train"] & by_partition["query"]
    train_gallery = by_partition["train"] & by_partition["gallery"]
    query_gallery = by_partition["query"] & by_partition["gallery"]

    print("\n[crop manifest raw-ID check]")
    print(f"  raw train ids:       {len(by_partition['train'])}")
    print(f"  raw query ids:       {len(by_partition['query'])}")
    print(f"  raw gallery ids:     {len(by_partition['gallery'])}")
    print(f"  raw train & query:   {len(train_query)}")
    print(f"  raw train & gallery: {len(train_gallery)}")
    print(f"  raw query & gallery: {len(query_gallery)}")
    if train_query or train_gallery:
        print(
            "  warning: raw identities overlap between train and evaluation splits. "
            "Standard train relabeling may hide this overlap."
        )
    for raw_identity, count in per_raw.most_common(5):
        print(f"  frequent raw id:     {raw_identity} -> {count} crops")


def main():
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts or [])
    root = args.root or cfg.DATASETS.ROOT_DIR

    print("=" * 80)
    print("MERGED REID DATASET AUDIT")
    print("=" * 80)
    print(f"Root: {root}")
    if not os.path.exists(root):
        print(f"  MISS {root}")
        print("\n[error] dataset root does not exist.")
        return 1

    missing_required = []
    for rel in ["bounding_box_train", "query", "bounding_box_test"]:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            missing_required.append(path)
        print(f"  {'OK ' if os.path.exists(path) else 'MISS'} {path}")
    for rel in ["bounding_box_train/opt", "bounding_box_train/sar"]:
        path = os.path.join(root, rel)
        print(f"  {'OK ' if os.path.exists(path) else 'SKIP'} {path} (optional; modality is parsed from filename)")

    if missing_required:
        print("\n[error] required ReID folders are missing:")
        for path in missing_required:
            print(f"  {path}")
        return 1

    dataset = MergedDataset(
        root=root,
        is_train=not args.eval_only,
        verbose=False,
        train_pair_only=cfg.DATASETS.TRAIN_PAIR_ONLY,
        pair_strategy=cfg.DATASETS.PAIR_STRATEGY,
        max_pair_per_id=cfg.DATASETS.MAX_PAIR_PER_ID,
    )

    if args.eval_only:
        train_ids, train_counts = set(), Counter()
        print("\n[train]")
        print("  skipped:    eval_only=True")
    else:
        train_ids, train_counts = summarize_split("train", dataset.train, dataset)
    query_ids, query_counts = summarize_split("query", dataset.query, dataset)
    gallery_ids, gallery_counts = summarize_split("gallery", dataset.gallery, dataset)
    if not args.eval_only:
        summarize_pairs(dataset)

    print("\n[ID intersections]")
    print_intersection("query & gallery", query_ids, gallery_ids)
    if not args.eval_only:
        print_intersection("train & query", train_ids, query_ids)
        print_intersection("train & gallery", train_ids, gallery_ids)

    missing_query = sorted(list(query_ids - gallery_ids))[:20]
    missing_gallery = sorted(list(gallery_ids - query_ids))[:20]
    if missing_query:
        print(f"  query ids absent from gallery, first 20: {missing_query}")
    if missing_gallery:
        print(f"  gallery ids absent from query, first 20: {missing_gallery}")

    if not args.eval_only:
        low_train_ids = [pid for pid, count in train_counts.items() if count < 2]
        if low_train_ids:
            print(f"\n[warning] train IDs with <2 images: {len(low_train_ids)}; first 20: {low_train_ids[:20]}")
    if len(query_ids & gallery_ids) == 0:
        print("\n[error] query and gallery have no overlapping IDs; ReID metrics are invalid.")
    if not args.eval_only and len(dataset.train_pair) == 0:
        print("\n[error] no opt/sar training pairs were found; cross-modal training is invalid.")
    summarize_crop_manifest(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
