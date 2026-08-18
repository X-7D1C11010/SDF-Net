import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOTATION_EXTS = {".txt", ".xml", ".json", ".geojson", ".mat", ".csv"}
DEFAULT_DATASETS = [
    "3MOS",
    "HOSS-ReID",
    "MOS-Ship",
    "Multi-Resolution-SAR-dataset",
    "OSdataset",
    "OSDataset2.0",
    "OsEval",
    "QXS-SAROPT",
]
OPT_TOKENS = {"opt", "optical", "rgb", "visible", "vis"}
SAR_TOKENS = {"sar", "radar"}
MODALITY_PATTERN = re.compile(
    r"(^|[_\-.])(optical|opt|rgb|visible|vis|sar|radar)(?=$|[_\-.])",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only inventory of raw optical/SAR datasets before rebuilding SDF-Net data."
    )
    parser.add_argument(
        "--root",
        default="/ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID",
        help="Root containing the original dataset directories.",
    )
    parser.add_argument(
        "--output_dir",
        default="./logs/dataset_rebuild_audit",
        help="Directory for CSV manifests and the text report.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset directory names to inspect.",
    )
    parser.add_argument(
        "--check_images",
        default=200,
        type=int,
        help="Number of images opened per dataset; use 0 to disable or -1 to check all.",
    )
    parser.add_argument(
        "--write_image_manifest",
        action="store_true",
        help="Write one CSV row per image. This can be large.",
    )
    parser.add_argument("--show_examples", default=10, type=int)
    return parser.parse_args()


def tokenize(value):
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def infer_modality(relative_path):
    tokens = set(tokenize(relative_path.as_posix()))
    has_opt = bool(tokens & OPT_TOKENS)
    has_sar = bool(tokens & SAR_TOKENS)
    if has_opt and has_sar:
        return "ambiguous"
    if has_opt:
        return "opt"
    if has_sar:
        return "sar"
    return "unknown"


def normalize_pair_component(component):
    if component.lower() in OPT_TOKENS | SAR_TOKENS:
        return ""
    cleaned = MODALITY_PATTERN.sub(lambda match: match.group(1), component.lower())
    cleaned = re.sub(r"[_\-.]+", "_", cleaned).strip("_")
    return cleaned


def candidate_pair_key(relative_path):
    without_suffix = relative_path.with_suffix("")
    parts = []
    for component in without_suffix.parts:
        normalized = normalize_pair_component(component)
        if normalized:
            parts.append(normalized)
    return "/".join(parts)


def select_for_check(paths, requested):
    if requested == 0 or not paths:
        return []
    if requested < 0 or requested >= len(paths):
        return paths
    if requested == 1:
        return [paths[0]]
    indices = {
        round(index * (len(paths) - 1) / (requested - 1))
        for index in range(requested)
    }
    return [paths[index] for index in sorted(indices)]


def verify_images(paths):
    from PIL import Image

    bad = []
    modes = Counter()
    sizes = Counter()
    for path in paths:
        try:
            with Image.open(path) as image:
                modes[image.mode] += 1
                sizes[f"{image.width}x{image.height}"] += 1
                image.verify()
        except Exception as exc:
            bad.append((str(path), str(exc)))
    return bad, modes, sizes


def inspect_dataset(dataset_name, dataset_root, check_images):
    extension_counts = Counter()
    modality_counts = Counter()
    annotation_counts = Counter()
    image_records = []
    access_errors = []
    total_files = 0
    total_bytes = 0

    def onerror(exc):
        access_errors.append(str(exc))

    for dirpath, _, filenames in os.walk(dataset_root, onerror=onerror, followlinks=False):
        for filename in filenames:
            total_files += 1
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            extension_counts[suffix or "<none>"] += 1
            try:
                size_bytes = path.stat().st_size
            except OSError as exc:
                size_bytes = 0
                access_errors.append(f"{path}: {exc}")
            total_bytes += size_bytes

            if suffix in ANNOTATION_EXTS:
                annotation_counts[suffix] += 1
            if suffix not in IMAGE_EXTS:
                continue

            relative_path = path.relative_to(dataset_root)
            modality = infer_modality(relative_path)
            modality_counts[modality] += 1
            image_records.append(
                {
                    "dataset": dataset_name,
                    "path": str(path),
                    "relative_path": relative_path.as_posix(),
                    "extension": suffix,
                    "size_bytes": size_bytes,
                    "modality": modality,
                    "candidate_pair_key": candidate_pair_key(relative_path),
                }
            )

    image_records.sort(key=lambda row: row["relative_path"])
    pair_groups = defaultdict(lambda: {"opt": [], "sar": []})
    for row in image_records:
        if row["modality"] in ("opt", "sar"):
            pair_groups[row["candidate_pair_key"]][row["modality"]].append(row["path"])

    candidate_pairs = []
    paired_keys = 0
    one_to_one_keys = 0
    ambiguous_pair_keys = 0
    opt_only_keys = 0
    sar_only_keys = 0
    for key, modalities in sorted(pair_groups.items()):
        opt_paths = modalities["opt"]
        sar_paths = modalities["sar"]
        if opt_paths and sar_paths:
            paired_keys += 1
            if len(opt_paths) == 1 and len(sar_paths) == 1:
                one_to_one_keys += 1
            else:
                ambiguous_pair_keys += 1
            candidate_pairs.append(
                {
                    "dataset": dataset_name,
                    "candidate_pair_key": key,
                    "opt_count": len(opt_paths),
                    "sar_count": len(sar_paths),
                    "opt_path": opt_paths[0],
                    "sar_path": sar_paths[0],
                    "pair_status": (
                        "one_to_one"
                        if len(opt_paths) == 1 and len(sar_paths) == 1
                        else "ambiguous"
                    ),
                }
            )
        elif opt_paths:
            opt_only_keys += 1
        elif sar_paths:
            sar_only_keys += 1

    paths_to_check = select_for_check(
        [Path(row["path"]) for row in image_records], check_images
    )
    bad_images, checked_modes, checked_sizes = verify_images(paths_to_check)

    summary = {
        "dataset": dataset_name,
        "root": str(dataset_root),
        "exists": True,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "image_count": len(image_records),
        "opt_images": modality_counts["opt"],
        "sar_images": modality_counts["sar"],
        "ambiguous_images": modality_counts["ambiguous"],
        "unknown_images": modality_counts["unknown"],
        "annotation_files": sum(annotation_counts.values()),
        "candidate_paired_keys": paired_keys,
        "candidate_one_to_one_keys": one_to_one_keys,
        "candidate_ambiguous_keys": ambiguous_pair_keys,
        "candidate_opt_only_keys": opt_only_keys,
        "candidate_sar_only_keys": sar_only_keys,
        "images_checked": len(paths_to_check),
        "bad_images": len(bad_images),
        "access_errors": len(access_errors),
        "extensions": dict(extension_counts),
        "annotation_extensions": dict(annotation_counts),
        "checked_modes": dict(checked_modes),
        "checked_sizes": dict(checked_sizes),
    }
    return summary, image_records, candidate_pairs, bad_images, access_errors


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def human_bytes(value):
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def main():
    args = parse_args()
    source_root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {source_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_image_records = []
    all_candidate_pairs = []
    all_bad_images = []
    all_access_errors = []

    for dataset_name in args.datasets:
        dataset_root = source_root / dataset_name
        if not dataset_root.is_dir():
            summaries.append(
                {
                    "dataset": dataset_name,
                    "root": str(dataset_root),
                    "exists": False,
                    "total_files": 0,
                    "total_bytes": 0,
                    "image_count": 0,
                    "opt_images": 0,
                    "sar_images": 0,
                    "ambiguous_images": 0,
                    "unknown_images": 0,
                    "annotation_files": 0,
                    "candidate_paired_keys": 0,
                    "candidate_one_to_one_keys": 0,
                    "candidate_ambiguous_keys": 0,
                    "candidate_opt_only_keys": 0,
                    "candidate_sar_only_keys": 0,
                    "images_checked": 0,
                    "bad_images": 0,
                    "access_errors": 0,
                    "extensions": {},
                    "annotation_extensions": {},
                    "checked_modes": {},
                    "checked_sizes": {},
                }
            )
            continue

        summary, records, pairs, bad_images, access_errors = inspect_dataset(
            dataset_name, dataset_root, args.check_images
        )
        summaries.append(summary)
        all_image_records.extend(records)
        all_candidate_pairs.extend(pairs)
        all_bad_images.extend(
            {"dataset": dataset_name, "path": path, "error": error}
            for path, error in bad_images
        )
        all_access_errors.extend(
            {"dataset": dataset_name, "error": error} for error in access_errors
        )

    summary_fields = [
        "dataset",
        "root",
        "exists",
        "total_files",
        "total_bytes",
        "image_count",
        "opt_images",
        "sar_images",
        "ambiguous_images",
        "unknown_images",
        "annotation_files",
        "candidate_paired_keys",
        "candidate_one_to_one_keys",
        "candidate_ambiguous_keys",
        "candidate_opt_only_keys",
        "candidate_sar_only_keys",
        "images_checked",
        "bad_images",
        "access_errors",
        "extensions",
        "annotation_extensions",
        "checked_modes",
        "checked_sizes",
    ]
    write_csv(output_dir / "dataset_summary.csv", summary_fields, summaries)
    write_csv(
        output_dir / "candidate_pairs.csv",
        [
            "dataset",
            "candidate_pair_key",
            "opt_count",
            "sar_count",
            "opt_path",
            "sar_path",
            "pair_status",
        ],
        all_candidate_pairs,
    )
    if args.write_image_manifest:
        write_csv(
            output_dir / "image_manifest.csv",
            [
                "dataset",
                "path",
                "relative_path",
                "extension",
                "size_bytes",
                "modality",
                "candidate_pair_key",
            ],
            all_image_records,
        )
    write_csv(output_dir / "bad_images.csv", ["dataset", "path", "error"], all_bad_images)
    write_csv(output_dir / "access_errors.csv", ["dataset", "error"], all_access_errors)

    lines = [
        "=" * 100,
        "RAW OPTICAL/SAR DATASET AUDIT",
        "=" * 100,
        f"Source root: {source_root}",
        f"Output dir:  {output_dir}",
        "",
        "Candidate pairs are filename/path heuristics only. Do not train from them before validation.",
    ]
    for summary in summaries:
        lines.extend(
            [
                "",
                f"[{summary['dataset']}]",
                f"  exists:                  {summary['exists']}",
                f"  total size:              {human_bytes(summary['total_bytes'])}",
                f"  files/images/labels:     {summary['total_files']} / {summary['image_count']} / {summary['annotation_files']}",
                f"  opt/sar/unknown/ambig:   {summary['opt_images']} / {summary['sar_images']} / {summary['unknown_images']} / {summary['ambiguous_images']}",
                f"  candidate paired keys:   {summary['candidate_paired_keys']}",
                f"  one-to-one/ambiguous:    {summary['candidate_one_to_one_keys']} / {summary['candidate_ambiguous_keys']}",
                f"  opt-only/sar-only keys:  {summary['candidate_opt_only_keys']} / {summary['candidate_sar_only_keys']}",
                f"  checked/bad images:      {summary['images_checked']} / {summary['bad_images']}",
                f"  access errors:           {summary['access_errors']}",
                f"  extensions:              {summary['extensions']}",
                f"  checked image modes:     {summary['checked_modes']}",
            ]
        )
    if all_bad_images:
        lines.append("\n[bad image examples]")
        for row in all_bad_images[: args.show_examples]:
            lines.append(f"  {row['dataset']}: {row['path']} ({row['error']})")
    if all_access_errors:
        lines.append("\n[access error examples]")
        for row in all_access_errors[: args.show_examples]:
            lines.append(f"  {row['dataset']}: {row['error']}")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    with open(output_dir / "audit.txt", "w", encoding="utf-8") as file_obj:
        file_obj.write(report)
    print(f"Reports written to: {output_dir}")
    return 1 if any(not row["exists"] for row in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
