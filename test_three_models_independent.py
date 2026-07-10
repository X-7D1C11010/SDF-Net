import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run independent-test cross-modal matching for multiple trained models."
    )
    parser.add_argument(
        "--config",
        default="configs/test_three_models_independent.json",
        help="JSON runner config.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a configured model weight is missing.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bool_flag(command, flag, enabled):
    if enabled:
        command.append(flag)


def build_command(config, model, save_path):
    command = [
        sys.executable,
        "test_cross_modal.py",
        "--config_file",
        model.get("config_file") or config["test_config"],
        "--weight_path",
        model["weight_path"],
        "--save_path",
        str(save_path),
        "--distance_metric",
        config.get("distance_metric", "cosine_distance"),
        "--classifier_type",
        config.get("classifier_type", "threshold"),
        "--threshold_strategy",
        config.get("threshold_strategy", "mad"),
        "--threshold_percentile",
        str(config.get("threshold_percentile", 95.0)),
        "--threshold_mad_scale",
        str(config.get("threshold_mad_scale", 3.0)),
        "--topk",
        str(config.get("topk", 10)),
    ]

    if config.get("seed") is not None:
        command.extend(["--seed", str(config["seed"])])
    if config.get("manual_threshold") is not None:
        command.extend(["--manual_threshold", str(config["manual_threshold"])])
    if config.get("csls_k") is not None:
        command.extend(["--csls_k", str(config["csls_k"])])
    if config.get("mutual_k") is not None:
        command.extend(["--mutual_k", str(config["mutual_k"])])

    bool_flag(command, "--require_mutual", config.get("require_mutual", False))
    bool_flag(command, "--supervised_matcher", config.get("supervised_matcher", False))
    bool_flag(command, "--save_matrices", config.get("save_matrices", False))

    if config.get("compare_metrics", False):
        command.append("--compare_metrics")
        metrics = config.get("comparison_metrics") or ["cosine_distance", "csls_similarity", "hybrid"]
        command.append("--comparison_metrics")
        command.extend(metrics)

    extra_opts = []
    extra_opts.extend(config.get("opts", []))
    extra_opts.extend(model.get("opts", []))
    command.extend(extra_opts)
    return command


def run_command(command, log_path, dry_run=False):
    command_text = " ".join(command)
    print(command_text)
    with open(log_path.with_suffix(".cmd.txt"), "w", encoding="utf-8") as f:
        f.write(command_text + "\n")

    if dry_run:
        return 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def load_single_metrics(save_path):
    metrics_path = save_path / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_result(model_name, status, save_path, metrics):
    row = {
        "model": model_name,
        "status": status,
        "save_path": str(save_path),
        "mAP": "",
        "rank_1": "",
        "rank_5": "",
        "rank_10": "",
        "threshold_top1": "",
        "threshold_top5": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "balanced_accuracy": "",
        "separation_score": "",
        "accepted_pairs": "",
        "avg_matches_per_query": "",
    }
    if not metrics:
        return row

    reid = metrics.get("reid", {})
    topk = metrics.get("threshold_topk", {})
    basic = metrics.get("basic", {})
    diagnostics = metrics.get("distance_diagnostics", {})
    matching = metrics.get("matching", {})
    row.update(
        {
            "mAP": reid.get("mAP", ""),
            "rank_1": reid.get("rank_1", ""),
            "rank_5": reid.get("rank_5", ""),
            "rank_10": reid.get("rank_10", ""),
            "threshold_top1": topk.get("top_1_accuracy", ""),
            "threshold_top5": topk.get("top_5_accuracy", ""),
            "precision": basic.get("precision", ""),
            "recall": basic.get("recall", ""),
            "f1": basic.get("f1", ""),
            "balanced_accuracy": basic.get("balanced_accuracy", ""),
            "separation_score": diagnostics.get("separation_score", ""),
            "accepted_pairs": matching.get("accepted_pairs", ""),
            "avg_matches_per_query": matching.get("avg_matches_per_query", ""),
        }
    )
    return row


def write_summary(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to: {json_path}")
    print(f"Summary written to: {csv_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "./logs/Independent_Test_ThreeModels"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_is_error = args.strict or not config.get("skip_missing", False)

    for model in config.get("models", []):
        name = model["name"]
        weight_path = Path(model["weight_path"])
        save_path = output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"Testing model: {name}")
        print("=" * 80)

        if not args.dry_run and not weight_path.exists():
            message = f"Missing model weight: {weight_path}"
            if missing_is_error:
                raise FileNotFoundError(message)
            print(f"SKIP: {message}")
            rows.append(summarize_result(name, "missing_weight", save_path, None))
            continue

        command = build_command(config, model, save_path)
        returncode = run_command(command, save_path / "stdout.log", dry_run=args.dry_run)
        status = "dry_run" if args.dry_run else ("ok" if returncode == 0 else f"failed:{returncode}")
        metrics = None if args.dry_run else load_single_metrics(save_path)
        rows.append(summarize_result(name, status, save_path, metrics))
        if returncode != 0 and missing_is_error:
            raise RuntimeError(f"Model '{name}' failed with return code {returncode}")

    if rows:
        write_summary(output_dir, rows)
        print("\n" + "=" * 80)
        print("THREE-MODEL INDEPENDENT TEST SUMMARY")
        print("=" * 80)
        for row in rows:
            print(
                f"{row['model']:<28} status={row['status']:<12} "
                f"mAP={row['mAP']} rank1={row['rank_1']} rank5={row['rank_5']} "
                f"thr_top1={row['threshold_top1']} thr_top5={row['threshold_top5']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
