import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a conservative pseudo opt-SAR pair manifest from top_matches.csv."
    )
    parser.add_argument("--top_matches", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--query_root",
        default=None,
        type=str,
        help="Optional root used to expand old CSV query filenames into full paths.",
    )
    parser.add_argument(
        "--gallery_root",
        default=None,
        type=str,
        help="Optional root used to expand old CSV gallery filenames into full paths.",
    )
    parser.add_argument("--max_rank", default=1, type=int)
    parser.add_argument(
        "--require_threshold",
        action="store_true",
        help="Keep only rows with accepted_by_threshold == 1.",
    )
    parser.add_argument(
        "--max_distance",
        default=None,
        type=float,
        help="Optional maximum ranking_distance.",
    )
    parser.add_argument(
        "--dedupe_gallery",
        action="store_true",
        help="Keep only the first query assigned to each gallery image.",
    )
    return parser.parse_args()


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def expand_path(path, root):
    if not path or os.path.isabs(path) or root is None:
        return path
    return os.path.join(root, path)


def main():
    args = parse_args()
    rows = []
    seen_queries = set()
    seen_gallery = set()

    with open(args.top_matches, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = as_int(row.get("rank"))
            if rank > args.max_rank:
                continue
            if args.require_threshold and as_int(row.get("accepted_by_threshold")) != 1:
                continue
            if args.max_distance is not None and as_float(row.get("ranking_distance")) > args.max_distance:
                continue

            q_path = expand_path(row.get("query_path"), args.query_root)
            g_path = expand_path(row.get("gallery_path"), args.gallery_root)
            if not q_path or not g_path:
                continue
            if q_path in seen_queries:
                continue
            if args.dedupe_gallery and g_path in seen_gallery:
                continue

            seen_queries.add(q_path)
            seen_gallery.add(g_path)
            rows.append(
                {
                    "opt_path": q_path,
                    "sar_path": g_path,
                    "query_index": row.get("query_index", ""),
                    "gallery_index": row.get("gallery_index", ""),
                    "ranking_distance": row.get("ranking_distance", ""),
                    "metric_value": row.get("metric_value", ""),
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "opt_path",
                "sar_path",
                "query_index",
                "gallery_index",
                "ranking_distance",
                "metric_value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("PSEUDO PAIR MANIFEST")
    print("=" * 80)
    print(f"Input:          {args.top_matches}")
    print(f"Pseudo pairs:   {len(rows)}")
    print(f"Output:         {args.output}")
    if len(rows) < 100:
        print("WARNING: very few pseudo pairs were kept. Use this only as a diagnostic or loosen filters.")


if __name__ == "__main__":
    main()
