import argparse
import csv
import os
from collections import Counter, defaultdict


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MODALITY_SUFFIXES = ("opt", "rgb", "visible", "vis", "sar", "ir")
DEFAULT_EXPECTED_SOURCES = [
    "3MOS",
    "HOSS-ReID",
    "OptiSar_Pair",
    "MOS-Ship",
    "Multi-Resolution-SAR-dataset",
    "OSdataset",
    "OSdataset2.0",
    "OsEval",
    "QXS-SAROPT",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit opt/SAR paired data used for first-stage cross-modal pretraining."
    )
    parser.add_argument(
        "--root",
        default="/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/Pretrain",
        type=str,
        help="Pretrain root containing opt/ and sar/ subfolders.",
    )
    parser.add_argument("--opt_dir", default=None, type=str)
    parser.add_argument("--sar_dir", default=None, type=str)
    parser.add_argument(
        "--expected_sources",
        nargs="*",
        default=DEFAULT_EXPECTED_SOURCES,
        help="Source dataset names expected to appear in paths or filenames.",
    )
    parser.add_argument(
        "--pair_key",
        default="relative",
        choices=["relative", "basename"],
        help=(
            "relative keeps subfolder context and is safer for merged multi-dataset "
            "pretraining; basename ignores subfolders."
        ),
    )
    parser.add_argument(
        "--write_manifest",
        default=None,
        type=str,
        help="Optional CSV path with opt_path,sar_path pairs for pretraining.",
    )
    parser.add_argument(
        "--duplicate_policy",
        default="first",
        choices=["first", "cartesian", "skip"],
        help="How manifest writing handles duplicate pair keys.",
    )
    parser.add_argument("--show_unmatched", default=20, type=int)
    parser.add_argument("--show_duplicates", default=20, type=int)
    parser.add_argument(
        "--check_images",
        action="store_true",
        help="Open a small sample of images to catch corrupt files.",
    )
    parser.add_argument("--max_check", default=200, type=int)
    return parser.parse_args()


def normalize_token(text):
    return "".join(ch for ch in text.lower() if ch.isalnum())


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


def collect_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(root, file_name))
    return sorted(paths)


def rel_path(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def make_pair_key(path, modality_root, mode):
    rel = rel_path(path, modality_root)
    stem = os.path.splitext(rel)[0]
    dirname = os.path.dirname(stem).replace("\\", "/")
    basename = os.path.basename(stem)
    basename = strip_modality_suffix(basename)
    if mode == "basename":
        return basename.lower()
    if dirname:
        return f"{dirname}/{basename}".lower()
    return basename.lower()


def detect_source(path, modality_root, expected_sources):
    rel = rel_path(path, modality_root)
    norm_rel = normalize_token(rel)
    norm_full = normalize_token(path)
    for source in expected_sources:
        norm_source = normalize_token(source)
        if norm_source and (norm_source in norm_rel or norm_source in norm_full):
            return source
    first_part = rel.split("/", 1)[0]
    if first_part != rel:
        return first_part
    return "unknown"


def build_records(paths, modality_root, expected_sources, pair_key_mode):
    records = []
    key_to_paths = defaultdict(list)
    source_counter = Counter()
    ext_counter = Counter()

    for path in paths:
        key = make_pair_key(path, modality_root, pair_key_mode)
        source = detect_source(path, modality_root, expected_sources)
        ext = os.path.splitext(path)[1].lower()
        record = {
            "path": path,
            "rel_path": rel_path(path, modality_root),
            "pair_key": key,
            "source": source,
            "ext": ext,
        }
        records.append(record)
        key_to_paths[key].append(record)
        source_counter[source] += 1
        ext_counter[ext] += 1

    return records, key_to_paths, source_counter, ext_counter


def duplicate_keys(key_to_paths):
    return {key: records for key, records in key_to_paths.items() if len(records) > 1}


def pair_rows(opt_by_key, sar_by_key, duplicate_policy):
    rows = []
    paired_keys = sorted(set(opt_by_key.keys()) & set(sar_by_key.keys()))
    skipped_duplicate_keys = []

    for key in paired_keys:
        opt_records = opt_by_key[key]
        sar_records = sar_by_key[key]
        has_duplicate = len(opt_records) > 1 or len(sar_records) > 1
        if has_duplicate and duplicate_policy == "skip":
            skipped_duplicate_keys.append(key)
            continue
        if has_duplicate and duplicate_policy == "cartesian":
            for opt_record in opt_records:
                for sar_record in sar_records:
                    rows.append(make_manifest_row(key, opt_record, sar_record))
        else:
            rows.append(make_manifest_row(key, opt_records[0], sar_records[0]))

    return rows, skipped_duplicate_keys


def make_manifest_row(key, opt_record, sar_record):
    return {
        "opt_path": opt_record["path"],
        "sar_path": sar_record["path"],
        "pair_key": key,
        "opt_source": opt_record["source"],
        "sar_source": sar_record["source"],
    }


def print_counter(title, counter):
    print(f"\n[{title}]")
    if not counter:
        print("  empty")
        return
    for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:<36} {count}")


def print_examples(title, items, limit):
    print(f"\n[{title}]")
    if not items:
        print("  none")
        return
    for item in list(items)[:limit]:
        print(f"  {item}")


def image_check(paths, max_check):
    from PIL import Image

    bad = []
    checked = 0
    for path in paths[:max_check]:
        checked += 1
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            bad.append((path, str(exc)))
    return checked, bad


def main():
    args = parse_args()
    opt_dir = args.opt_dir or os.path.join(args.root, "opt")
    sar_dir = args.sar_dir or os.path.join(args.root, "sar")

    if not os.path.isdir(opt_dir):
        raise FileNotFoundError(f"OPT directory not found: {opt_dir}")
    if not os.path.isdir(sar_dir):
        raise FileNotFoundError(f"SAR directory not found: {sar_dir}")

    opt_paths = collect_images(opt_dir)
    sar_paths = collect_images(sar_dir)

    opt_records, opt_by_key, opt_sources, opt_exts = build_records(
        opt_paths, opt_dir, args.expected_sources, args.pair_key
    )
    sar_records, sar_by_key, sar_sources, sar_exts = build_records(
        sar_paths, sar_dir, args.expected_sources, args.pair_key
    )

    opt_keys = set(opt_by_key.keys())
    sar_keys = set(sar_by_key.keys())
    paired_keys = opt_keys & sar_keys
    opt_only = sorted(opt_keys - sar_keys)
    sar_only = sorted(sar_keys - opt_keys)
    opt_dupes = duplicate_keys(opt_by_key)
    sar_dupes = duplicate_keys(sar_by_key)
    rows, skipped_duplicate_keys = pair_rows(opt_by_key, sar_by_key, args.duplicate_policy)

    seen_sources = set(opt_sources.keys()) | set(sar_sources.keys())
    expected_missing = [
        source for source in args.expected_sources if source not in seen_sources
    ]

    print("=" * 80)
    print("CROSS-MODAL PRETRAIN DATASET AUDIT")
    print("=" * 80)
    print(f"Root:              {args.root}")
    print(f"OPT dir:           {opt_dir}")
    print(f"SAR dir:           {sar_dir}")
    print(f"Pair key mode:     {args.pair_key}")
    print(f"OPT images:        {len(opt_paths)}")
    print(f"SAR images:        {len(sar_paths)}")
    print(f"OPT unique keys:   {len(opt_keys)}")
    print(f"SAR unique keys:   {len(sar_keys)}")
    print(f"Paired keys:       {len(paired_keys)}")
    print(f"Manifest rows:     {len(rows)}")
    print(f"OPT-only keys:     {len(opt_only)}")
    print(f"SAR-only keys:     {len(sar_only)}")
    print(f"OPT duplicate keys:{len(opt_dupes)}")
    print(f"SAR duplicate keys:{len(sar_dupes)}")
    print(f"Missing expected:  {expected_missing if expected_missing else 'none'}")

    print_counter("OPT source counts", opt_sources)
    print_counter("SAR source counts", sar_sources)
    print_counter("OPT extension counts", opt_exts)
    print_counter("SAR extension counts", sar_exts)

    print_examples("OPT-only pair keys", opt_only, args.show_unmatched)
    print_examples("SAR-only pair keys", sar_only, args.show_unmatched)
    print_examples("OPT duplicate pair keys", opt_dupes.keys(), args.show_duplicates)
    print_examples("SAR duplicate pair keys", sar_dupes.keys(), args.show_duplicates)

    if skipped_duplicate_keys:
        print_examples(
            "Skipped duplicate pair keys for manifest",
            skipped_duplicate_keys,
            args.show_duplicates,
        )

    if args.check_images:
        checked_opt, bad_opt = image_check(opt_paths, args.max_check)
        checked_sar, bad_sar = image_check(sar_paths, args.max_check)
        print("\n[image integrity sample]")
        print(f"  checked opt: {checked_opt}, bad: {len(bad_opt)}")
        print(f"  checked sar: {checked_sar}, bad: {len(bad_sar)}")
        for path, error in bad_opt[:10] + bad_sar[:10]:
            print(f"  BAD {path}: {error}")

    if args.write_manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.write_manifest)), exist_ok=True)
        with open(args.write_manifest, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["opt_path", "sar_path", "pair_key", "opt_source", "sar_source"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nManifest saved to: {args.write_manifest}")

    if len(paired_keys) == 0:
        print("\n[error] no opt/SAR paired keys were found.")
    elif expected_missing:
        print("\n[warning] some expected sources were not detected; inspect paths or source names.")
    elif opt_dupes or sar_dupes:
        print("\n[warning] duplicate pair keys exist; prefer --pair_key relative and inspect duplicates.")
    else:
        print("\nOK: paired pretraining data looks usable.")


if __name__ == "__main__":
    main()
