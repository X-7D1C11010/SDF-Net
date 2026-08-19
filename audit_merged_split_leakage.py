import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit source-level split leakage and cross-modal evaluation coverage in a merged ReID dataset."
    )
    parser.add_argument("--root", required=True, help="Merged dataset root containing merge_manifest.csv.")
    return parser.parse_args()


def set_intersection(left, right):
    return sorted(left & right)


def main():
    args = parse_args()
    root = Path(args.root)
    manifest_path = root / "merge_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    by_source_split_pid = defaultdict(lambda: defaultdict(set))
    by_source_split_old_pid = defaultdict(lambda: defaultdict(set))
    by_source_split_pid_modality = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    by_source_split_old_pid_modality = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    examples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = defaultdict(Counter)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"source_name", "split", "old_pid", "new_pid", "modality"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"merge_manifest.csv is missing columns: {sorted(missing)}")
        for row in reader:
            source = row["source_name"]
            split = row["split"]
            old_pid = row["old_pid"]
            new_pid = row["new_pid"]
            modality = row["modality"]
            by_source_split_pid[source][split].add(new_pid)
            by_source_split_old_pid[source][split].add(old_pid)
            by_source_split_pid_modality[source][split][modality].add(new_pid)
            by_source_split_old_pid_modality[source][split][modality].add(old_pid)
            if len(examples[source][split][old_pid]) < 3:
                examples[source][split][old_pid].append(row.get("source_path", ""))
            counts[(source, split)]["images"] += 1
            counts[(source, split)]["pids"] = len(by_source_split_pid[source][split])

    print("=" * 80)
    print("MERGED REID SPLIT LEAKAGE AUDIT")
    print("=" * 80)
    print(f"Root: {root}")

    total_train_query = set()
    total_train_gallery = set()
    total_raw_train_query = set()
    total_raw_train_gallery = set()
    for source in sorted(by_source_split_pid):
        splits = by_source_split_pid[source]
        old_splits = by_source_split_old_pid[source]
        train = splits["train"]
        query = splits["query"]
        gallery = splits["gallery"]
        old_train = old_splits["train"]
        old_query = old_splits["query"]
        old_gallery = old_splits["gallery"]
        train_query = set_intersection(train, query)
        train_gallery = set_intersection(train, gallery)
        query_gallery = set_intersection(query, gallery)
        old_train_query = set_intersection(old_train, old_query)
        old_train_gallery = set_intersection(old_train, old_gallery)
        old_query_gallery = set_intersection(old_query, old_gallery)
        total_train_query.update(train_query)
        total_train_gallery.update(train_gallery)
        total_raw_train_query.update((source, pid) for pid in old_train_query)
        total_raw_train_gallery.update((source, pid) for pid in old_train_gallery)

        print(f"\n[{source}]")
        for split in ("train", "query", "gallery"):
            print(
                f"  {split:<8} images={counts[(source, split)]['images']:<6} "
                f"ids={len(splits[split])}"
            )
        print(f"  train & query:   {len(train_query)}")
        print(f"  train & gallery: {len(train_gallery)}")
        print(f"  query & gallery: {len(query_gallery)}")
        print("  [raw old_pid, before merge renumbering]")
        print(f"  train & query:   {len(old_train_query)}")
        print(f"  train & gallery: {len(old_train_gallery)}")
        print(f"  query & gallery: {len(old_query_gallery)}")
        if old_train_query:
            print(f"  leaked raw train/query PIDs: {old_train_query[:20]}")
            for pid in old_train_query[:5]:
                print(f"    raw PID {pid} train: {examples[source]['train'][pid][:2]}")
                print(f"    raw PID {pid} query: {examples[source]['query'][pid][:2]}")
        if old_train_gallery:
            print(f"  leaked raw train/gallery PIDs: {old_train_gallery[:20]}")
            for pid in old_train_gallery[:5]:
                print(f"    raw PID {pid} train: {examples[source]['train'][pid][:2]}")
                print(f"    raw PID {pid} gallery: {examples[source]['gallery'][pid][:2]}")
        if train_query:
            print(f"  leaked train/query PIDs: {train_query[:20]}")
        if train_gallery:
            print(f"  leaked train/gallery PIDs: {train_gallery[:20]}")

        q_opt = by_source_split_pid_modality[source]["query"]["opt"]
        q_sar = by_source_split_pid_modality[source]["query"]["sar"]
        g_opt = by_source_split_pid_modality[source]["gallery"]["opt"]
        g_sar = by_source_split_pid_modality[source]["gallery"]["sar"]
        print(f"  opt query -> sar gallery IDs: {len(q_opt & g_sar)}")
        print(f"  sar query -> opt gallery IDs: {len(q_sar & g_opt)}")

    print("\n[overall]")
    print(f"  train & query:   {len(total_train_query)}")
    print(f"  train & gallery: {len(total_train_gallery)}")
    print("  [raw old_pid across sources]")
    print(f"  train & query:   {len(total_raw_train_query)}")
    print(f"  train & gallery: {len(total_raw_train_gallery)}")
    if total_train_query or total_train_gallery or total_raw_train_query or total_raw_train_gallery:
        print("  FAIL: do not train or report final metrics until split leakage is fixed.")
        return 2
    print("  OK: merged train and evaluation PID spaces are disjoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
