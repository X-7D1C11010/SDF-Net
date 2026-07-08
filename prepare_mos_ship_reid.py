import argparse
import csv
import os
from collections import Counter, defaultdict

from PIL import Image, ImageDraw


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
OPT_MODALITIES = {"rgb", "opt", "visible", "vis", "rgb_clouds"}
SAR_MODALITIES = {"sar", "ir"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop MOS-Ship oriented detection annotations into ReID-style "
            "bounding_box_train/query/bounding_box_test folders."
        )
    )
    parser.add_argument(
        "--src_root",
        default="/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/MOS-Ship/MOS-Ship",
        type=str,
        help="MOS-Ship root containing train/ and val/.",
    )
    parser.add_argument(
        "--dst_root",
        required=True,
        type=str,
        help="Output ReID root. Original MOS-Ship data is never modified.",
    )
    parser.add_argument("--pid_start", default=1015, type=int)
    parser.add_argument(
        "--identity_mode",
        default="annotation_id",
        choices=["annotation_id", "file_line"],
        help=(
            "annotation_id uses the last annotation field as ReID identity. "
            "file_line treats every annotation as a separate identity."
        ),
    )
    parser.add_argument(
        "--crop_mode",
        default="aabb_mask",
        choices=["aabb", "aabb_mask"],
        help="aabb crops the axis-aligned envelope; aabb_mask masks outside the quadrilateral.",
    )
    parser.add_argument("--padding", default=8, type=int)
    parser.add_argument("--min_size", default=8, type=int)
    parser.add_argument(
        "--include_rgb_clouds",
        action="store_true",
        help="Also crop val/rgb_clouds as opt samples using rgb_labelTxt.",
    )
    parser.add_argument(
        "--require_pair_for_train",
        action="store_true",
        help="Keep only train identities that have both opt and sar samples.",
    )
    parser.add_argument(
        "--require_pair_for_eval",
        action="store_true",
        help="Keep only val identities that have both opt query and sar gallery samples.",
    )
    parser.add_argument(
        "--copy_train_to_eval_if_missing",
        action="store_true",
        help=(
            "If val has no valid opt/SAR paired identities, split train paired identities "
            "into train/query/gallery. Use only for sanity checks, not final evaluation."
        ),
    )
    parser.add_argument("--eval_ratio", default=0.2, type=float)
    parser.add_argument("--camera_token", default="c1", type=str)
    parser.add_argument("--image_id_start", default=1, type=int)
    parser.add_argument("--image_id_width", default=0, type=int)
    parser.add_argument("--output_ext", default=".png", choices=[".png", ".jpg", ".jpeg"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def collect_image_index(root):
    index = defaultdict(list)
    if not root or not os.path.isdir(root):
        return index
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTS):
                stem = os.path.splitext(filename)[0].lower()
                index[stem].append(os.path.join(dirpath, filename))
    return index


def find_image(stem, image_indices, modality):
    candidates = []
    clean_stem = stem.lower()
    trial_stems = [clean_stem]
    if modality == "rgb_clouds":
        trial_stems.extend(
            [
                clean_stem.replace("_rgb", "_rgb_clouds"),
                clean_stem.replace("_opt", "_rgb_clouds"),
            ]
        )

    for trial in trial_stems:
        for image_index in image_indices:
            candidates.extend(image_index.get(trial, []))
    if candidates:
        return sorted(candidates)[0]
    return None


def parse_modality_from_stem(stem, fallback=None):
    lower = stem.lower()
    parts = lower.split("_")
    if parts:
        tail = parts[-1]
        if tail in OPT_MODALITIES:
            return "opt"
        if tail in SAR_MODALITIES:
            return "sar"
    if fallback in OPT_MODALITIES:
        return "opt"
    if fallback in SAR_MODALITIES:
        return "sar"
    return None


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    try:
        coords = [float(value) for value in parts[:8]]
    except ValueError:
        return None
    return {
        "coords": coords,
        "class_name": parts[8],
        "object_id": parts[9],
        "extra": parts[10:],
    }


def read_annotations(label_path):
    records = []
    with open(label_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            parsed["line_no"] = line_no
            records.append(parsed)
    return records


def numeric_key(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def make_raw_identity(annotation, label_stem, identity_mode):
    if identity_mode == "annotation_id":
        return annotation["object_id"]
    return f"{label_stem}:{annotation['line_no']}"


def clip_box(coords, width, height, padding):
    xs = coords[0::2]
    ys = coords[1::2]
    left = max(0, int(min(xs) - padding))
    top = max(0, int(min(ys) - padding))
    right = min(width, int(max(xs) + padding + 1))
    bottom = min(height, int(max(ys) + padding + 1))
    return left, top, right, bottom


def crop_annotation(image_path, coords, crop_mode, padding, min_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left, top, right, bottom = clip_box(coords, width, height, padding)
    if right - left < min_size or bottom - top < min_size:
        return None, None

    crop = image.crop((left, top, right, bottom))
    if crop_mode == "aabb_mask":
        local_points = [
            (coords[i] - left, coords[i + 1] - top)
            for i in range(0, len(coords), 2)
        ]
        mask = Image.new("L", crop.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(local_points, fill=255)
        background = Image.new("RGB", crop.size, (0, 0, 0))
        crop = Image.composite(crop, background, mask)

    return crop, (left, top, right, bottom)


def iter_label_files(folder):
    if not os.path.isdir(folder):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def collect_records(args):
    src_root = args.src_root

    train_images = collect_image_index(os.path.join(src_root, "train", "images"))
    val_rgb = collect_image_index(os.path.join(src_root, "val", "rgb"))
    val_rgb_clouds = collect_image_index(os.path.join(src_root, "val", "rgb_clouds"))
    val_sar = collect_image_index(os.path.join(src_root, "val", "sar"))

    label_specs = [
        {
            "partition": "train",
            "label_dir": os.path.join(src_root, "train", "labelTxt"),
            "image_indices": [train_images],
            "fallback_modality": None,
        },
        {
            "partition": "query",
            "label_dir": os.path.join(src_root, "val", "rgb_labelTxt"),
            "image_indices": [val_rgb],
            "fallback_modality": "rgb",
        },
        {
            "partition": "gallery",
            "label_dir": os.path.join(src_root, "val", "sar_labelTxt"),
            "image_indices": [val_sar],
            "fallback_modality": "sar",
        },
    ]
    if args.include_rgb_clouds:
        label_specs.append(
            {
                "partition": "query",
                "label_dir": os.path.join(src_root, "val", "rgb_labelTxt"),
                "image_indices": [val_rgb_clouds],
                "fallback_modality": "rgb_clouds",
            }
        )

    records = []
    missing_images = []
    malformed = 0

    for spec in label_specs:
        for label_path in iter_label_files(spec["label_dir"]):
            label_stem = os.path.splitext(os.path.basename(label_path))[0]
            modality = parse_modality_from_stem(label_stem, spec["fallback_modality"])
            image_path = find_image(label_stem, spec["image_indices"], spec["fallback_modality"])
            if image_path is None:
                missing_images.append(label_path)
                continue
            if modality is None:
                modality = "sar" if spec["fallback_modality"] == "sar" else "opt"

            annotations = read_annotations(label_path)
            if not annotations:
                malformed += 1
                continue
            for annotation in annotations:
                raw_identity = make_raw_identity(annotation, label_stem, args.identity_mode)
                records.append(
                    {
                        "partition": spec["partition"],
                        "label_path": label_path,
                        "label_stem": label_stem,
                        "image_path": image_path,
                        "modality": modality,
                        "class_name": annotation["class_name"],
                        "object_id": annotation["object_id"],
                        "raw_identity": raw_identity,
                        "line_no": annotation["line_no"],
                        "coords": annotation["coords"],
                    }
                )

    return records, missing_images, malformed


def paired_identity_sets(records):
    by_partition = defaultdict(lambda: defaultdict(set))
    for record in records:
        by_partition[record["partition"]][record["raw_identity"]].add(record["modality"])

    train_paired = {
        raw_id
        for raw_id, modalities in by_partition["train"].items()
        if {"opt", "sar"}.issubset(modalities)
    }
    eval_paired = {
        raw_id
        for raw_id in set(by_partition["query"].keys()) | set(by_partition["gallery"].keys())
        if "opt" in by_partition["query"].get(raw_id, set())
        and "sar" in by_partition["gallery"].get(raw_id, set())
    }
    all_paired = {
        raw_id
        for raw_id, modalities in merge_identity_modalities(records).items()
        if {"opt", "sar"}.issubset(modalities)
    }
    return train_paired, eval_paired, all_paired


def merge_identity_modalities(records):
    merged = defaultdict(set)
    for record in records:
        merged[record["raw_identity"]].add(record["modality"])
    return merged


def filter_records(args, records):
    train_paired, eval_paired, all_paired = paired_identity_sets(records)
    filtered = []
    for record in records:
        keep = True
        if record["partition"] == "train" and args.require_pair_for_train:
            keep = record["raw_identity"] in train_paired
        if record["partition"] in ("query", "gallery") and args.require_pair_for_eval:
            keep = record["raw_identity"] in eval_paired
        if keep:
            filtered.append(record)

    if args.copy_train_to_eval_if_missing:
        has_eval_pairs = bool(eval_paired)
        if not has_eval_pairs:
            filtered = split_train_to_eval(args, filtered, train_paired or all_paired)
    return filtered


def split_train_to_eval(args, records, paired_ids):
    paired_ids = sorted(paired_ids, key=numeric_key)
    eval_count = max(1, int(len(paired_ids) * args.eval_ratio)) if paired_ids else 0
    eval_ids = set(paired_ids[-eval_count:])
    output = []
    for record in records:
        new_record = dict(record)
        if record["partition"] == "train" and record["raw_identity"] in eval_ids:
            new_record["partition"] = "query" if record["modality"] == "opt" else "gallery"
        output.append(new_record)
    return output


def make_pid_map(records, pid_start):
    raw_ids = sorted({record["raw_identity"] for record in records}, key=numeric_key)
    return {raw_id: pid_start + idx for idx, raw_id in enumerate(raw_ids)}


def image_id_string(seq, width):
    if width and width > 0:
        return str(seq).zfill(width)
    return str(seq)


def partition_to_folder(partition):
    if partition == "train":
        return "bounding_box_train"
    if partition == "query":
        return "query"
    if partition == "gallery":
        return "bounding_box_test"
    raise ValueError(f"Unknown partition: {partition}")


def ensure_output_dirs(dst_root, overwrite):
    if os.path.exists(dst_root) and not overwrite:
        pass
    for folder in ("bounding_box_train", "query", "bounding_box_test"):
        os.makedirs(os.path.join(dst_root, folder), exist_ok=True)


def write_outputs(args, records):
    ensure_output_dirs(args.dst_root, args.overwrite)
    pid_map = make_pid_map(records, args.pid_start)
    manifest_rows = []
    image_seq = args.image_id_start
    skipped_small = 0
    skipped_existing = 0
    written = 0

    for record in sorted(
        records,
        key=lambda item: (
            item["partition"],
            numeric_key(item["raw_identity"]),
            item["image_path"],
            item["line_no"],
        ),
    ):
        pid = pid_map[record["raw_identity"]]
        img_id = image_id_string(image_seq, args.image_id_width)
        filename = f"{pid}_s{img_id}{args.camera_token}_{record['modality']}{args.output_ext}"
        out_dir = os.path.join(args.dst_root, partition_to_folder(record["partition"]))
        out_path = os.path.join(out_dir, filename)

        crop, crop_box = crop_annotation(
            record["image_path"],
            record["coords"],
            args.crop_mode,
            args.padding,
            args.min_size,
        )
        if crop is None:
            skipped_small += 1
            continue
        if os.path.exists(out_path) and not args.overwrite:
            skipped_existing += 1
            continue

        if not args.dry_run:
            crop.save(out_path)
        written += 1
        image_seq += 1
        manifest_rows.append(
            {
                "output_path": out_path,
                "pid": pid,
                "raw_identity": record["raw_identity"],
                "object_id": record["object_id"],
                "partition": record["partition"],
                "modality": record["modality"],
                "class_name": record["class_name"],
                "source_image": record["image_path"],
                "source_label": record["label_path"],
                "line_no": record["line_no"],
                "coords": " ".join(f"{value:.4f}" for value in record["coords"]),
                "crop_box": "" if crop_box is None else " ".join(str(v) for v in crop_box),
            }
        )

    if not args.dry_run:
        manifest_path = os.path.join(args.dst_root, "crop_manifest.csv")
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "output_path",
                    "pid",
                    "raw_identity",
                    "object_id",
                    "partition",
                    "modality",
                    "class_name",
                    "source_image",
                    "source_label",
                    "line_no",
                    "coords",
                    "crop_box",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        mapping_path = os.path.join(args.dst_root, "pid_mapping.csv")
        with open(mapping_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["raw_identity", "pid"])
            writer.writeheader()
            for raw_id, pid in sorted(pid_map.items(), key=lambda item: item[1]):
                writer.writerow({"raw_identity": raw_id, "pid": pid})

    return {
        "written": written,
        "skipped_small": skipped_small,
        "skipped_existing": skipped_existing,
        "pid_count": len(pid_map),
    }


def print_record_summary(records, title):
    print(f"\n[{title}]")
    print(f"  records: {len(records)}")
    print(f"  identities: {len({record['raw_identity'] for record in records})}")
    print(f"  partitions: {dict(Counter(record['partition'] for record in records))}")
    print(f"  modalities: {dict(Counter(record['modality'] for record in records))}")
    partition_modality = Counter(
        (record["partition"], record["modality"]) for record in records
    )
    for key, count in sorted(partition_modality.items()):
        print(f"  {key[0]:<8} {key[1]:<4}: {count}")


def main():
    args = parse_args()
    records, missing_images, malformed = collect_records(args)
    train_paired, eval_paired, all_paired = paired_identity_sets(records)
    filtered = filter_records(args, records)

    print("=" * 80)
    print("MOS-SHIP TO REID CROP")
    print("=" * 80)
    print(f"Source root:       {args.src_root}")
    print(f"Output root:       {args.dst_root}")
    print(f"Identity mode:     {args.identity_mode}")
    print(f"PID start:         {args.pid_start}")
    print(f"Crop mode:         {args.crop_mode}")
    print(f"Include clouds:    {args.include_rgb_clouds}")
    print(f"Dry run:           {args.dry_run}")
    print(f"Missing images:    {len(missing_images)}")
    print(f"Malformed labels:  {malformed}")
    print(f"Train paired IDs:  {len(train_paired)}")
    print(f"Eval paired IDs:   {len(eval_paired)}")
    print(f"All paired IDs:    {len(all_paired)}")
    if missing_images:
        print("  Missing image examples:")
        for path in missing_images[:20]:
            print(f"    {path}")

    print_record_summary(records, "raw annotation records")
    print_record_summary(filtered, "records after filters")

    result = write_outputs(args, filtered)
    print("\n[output]")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if args.identity_mode == "annotation_id" and len(all_paired) == 0:
        print(
            "\n[warning] no annotation IDs have both opt and sar samples. "
            "The last annotation field may not be a cross-modal identity ID; "
            "do not use this output as second-stage ReID training data."
        )


if __name__ == "__main__":
    main()
