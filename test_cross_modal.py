import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader
from utils.cross_modal_matching import CrossModalMatchingPipeline
from utils.progress_bar import MultiStepProgress, SimpleProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description="Label-independent cross-modal ReID testing")
    parser.add_argument(
        "--config_file",
        default="configs/SDF-Net_Test.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--distance_metric",
        default="cosine_distance",
        choices=[
            "euclidean",
            "cosine_distance",
            "cosine_similarity",
            "manhattan",
            "chebyshev",
            "minkowski",
            "mahalanobis",
            "hybrid",
        ],
        help="feature metric used for matching",
        type=str,
    )
    parser.add_argument(
        "--classifier_type",
        default="threshold",
        choices=[
            "threshold",
            "svm",
            "logistic_regression",
            "knn",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
        ],
        help="matcher type; threshold is label-independent by default",
        type=str,
    )
    parser.add_argument(
        "--threshold_strategy",
        default="mad",
        choices=["manual", "mad", "percentile", "otsu"],
        help="adaptive threshold strategy for threshold matcher",
        type=str,
    )
    parser.add_argument(
        "--threshold_percentile",
        default=95.0,
        help="acceptance percentile for nearest-neighbor based thresholding",
        type=float,
    )
    parser.add_argument(
        "--threshold_mad_scale",
        default=3.0,
        help="MAD multiplier for robust adaptive thresholding",
        type=float,
    )
    parser.add_argument(
        "--manual_threshold",
        default=None,
        help="manual threshold; enables manual threshold strategy",
        type=float,
    )
    parser.add_argument(
        "--require_mutual",
        action="store_true",
        help="keep only thresholded pairs that are mutual top-1 matches",
    )
    parser.add_argument(
        "--supervised_matcher",
        action="store_true",
        help="use query/gallery labels to fit matcher; intended only for ablation",
    )
    parser.add_argument(
        "--compare_metrics",
        action="store_true",
        help="compare multiple distance metrics with the same extracted features",
    )
    parser.add_argument(
        "--topk",
        default=10,
        help="number of ranked gallery candidates saved per query",
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="random seed used before model initialization and feature extraction",
        type=int,
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="path to save metrics and top-k matches",
        type=str,
    )
    parser.add_argument(
        "--save_matrices",
        action="store_true",
        help="save full metric/distance/match .npy matrices; this can require several GB",
    )
    parser.add_argument(
        "--weight_path",
        default=None,
        help="path to model weights; defaults to cfg.TEST.WEIGHT",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using command-line KEY VALUE pairs",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def configure(args):
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts or [])
    cfg.freeze()

    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_query_gallery(features, pids, camids, paths, num_query):
    q_features = features[:num_query]
    q_pids = pids[:num_query]
    q_camids = camids[:num_query]
    q_paths = paths[:num_query]

    g_features = features[num_query:]
    g_pids = pids[num_query:]
    g_camids = camids[num_query:]
    g_paths = paths[num_query:]
    return q_features, q_pids, q_camids, q_paths, g_features, g_pids, g_camids, g_paths


def matcher_kwargs(args):
    threshold = args.manual_threshold
    strategy = "manual" if threshold is not None else args.threshold_strategy
    return {
        "threshold_strategy": strategy,
        "threshold_percentile": args.threshold_percentile,
        "threshold_mad_scale": args.threshold_mad_scale,
        "threshold": threshold,
        "require_mutual": args.require_mutual,
    }


def run_single_metric(args, pipeline, data, progress, save_path):
    q_features, q_pids, q_camids, q_paths, g_features, g_pids, g_camids, g_paths = data

    progress.start_step(
        2,
        f"Calibrate {args.classifier_type} matcher with {args.distance_metric}",
    )
    start = time.time()
    pipeline.setup_matcher(
        distance_metric=args.distance_metric,
        classifier_type=args.classifier_type,
        **matcher_kwargs(args),
    )
    calibration = pipeline.calibrate_matcher(
        q_features,
        g_features,
        q_pids=q_pids,
        g_pids=g_pids,
        supervised=args.supervised_matcher,
    )
    progress.complete_step(2, f"calibration={calibration}, time={time.time() - start:.1f}s")

    progress.start_step(3, f"Compute {q_features.shape[0]}x{g_features.shape[0]} metric matrix")
    progress.start_step(4, "Compute matching and ReID metrics")
    results = pipeline.evaluate(
        q_features,
        g_features,
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
        q_paths=q_paths,
        g_paths=g_paths,
        topk=args.topk,
        save_path=save_path,
        save_matrices=args.save_matrices,
    )
    progress.complete_step(3, "metric matrix ready")
    progress.complete_step(4, "metrics ready")

    progress.start_step(5, "Print and save summary")
    pipeline.print_results()
    progress.complete_step(5, f"saved to {save_path}")
    return results


def run_metric_comparison(args, pipeline, data, save_path):
    q_features, q_pids, q_camids, q_paths, g_features, g_pids, g_camids, g_paths = data
    metrics_to_compare = ["cosine_distance", "euclidean", "hybrid", "manhattan"]
    results = {}

    for idx, metric in enumerate(metrics_to_compare, start=1):
        print(f"\n[{idx}/{len(metrics_to_compare)}] Comparing metric: {metric}")
        sys.stdout.flush()
        pipeline.setup_matcher(
            distance_metric=metric,
            classifier_type=args.classifier_type,
            **matcher_kwargs(args),
        )
        pipeline.calibrate_matcher(
            q_features,
            g_features,
            q_pids=q_pids,
            g_pids=g_pids,
            supervised=args.supervised_matcher,
        )
        metric_save_path = os.path.join(save_path, metric) if save_path else None
        results[metric] = pipeline.evaluate(
            q_features,
            g_features,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            q_paths=q_paths,
            g_paths=g_paths,
            topk=args.topk,
            save_path=metric_save_path,
            save_matrices=args.save_matrices,
        )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "comparison_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"{'Metric':<20} {'mAP':<10} {'Rank-1':<10} "
        f"{'ThrTop1':<10} {'ThrTop5':<10} {'Precision':<10} {'F1':<10}"
    )
    for metric, result in results.items():
        threshold_topk = result.get("threshold_topk", {})
        print(
            f"{metric:<20} "
            f"{result['reid'].get('mAP', 0):<10.4f} "
            f"{result['reid'].get('rank_1', 0):<10.4f} "
            f"{threshold_topk.get('top_1_accuracy', 0):<10.4f} "
            f"{threshold_topk.get('top_5_accuracy', 0):<10.4f} "
            f"{result['basic'].get('precision', 0):<10.4f} "
            f"{result['basic'].get('f1', 0):<10.4f}"
        )
    return results


def main():
    args = parse_args()
    configure(args)
    seed = args.seed if args.seed is not None else cfg.SOLVER.SEED
    set_seed(seed)
    print(f"Random seed: {seed}")

    if args.classifier_type != "threshold" and not args.supervised_matcher:
        raise ValueError(
            "Non-threshold classifiers require --supervised_matcher because they need labels. "
            "Use the default threshold matcher for label-independent testing."
        )
    if args.threshold_strategy == "manual" and args.manual_threshold is None:
        raise ValueError("--threshold_strategy manual requires --manual_threshold.")

    progress = MultiStepProgress(
        total_steps=6,
        step_names=[
            "Load data and model",
            "Extract features",
            "Calibrate matcher",
            "Compute distance matrix",
            "Compute metrics",
            "Save and print results",
        ],
    )

    progress.start_step(0, "Create dataloader and initialize SDF-Net feature extractor")
    print("Creating dataloaders...")
    sys.stdout.flush()
    (
        _train_loader,
        _train_loader_normal,
        val_loader,
        num_query,
        _num_classes,
        _camera_num,
    ) = make_dataloader(cfg, is_train=False)

    print("Dataset statistics:")
    print(f"  Query samples:   {num_query}")
    print(f"  Gallery samples: {len(val_loader.dataset) - num_query}")
    model_weight_path = args.weight_path if args.weight_path else cfg.TEST.WEIGHT
    print(f"Model weights: {model_weight_path}")
    pipeline = CrossModalMatchingPipeline(cfg, model_weight_path)
    progress.complete_step(0, f"query={num_query}, gallery={len(val_loader.dataset) - num_query}")

    progress.start_step(1, f"Extract normalized features from {len(val_loader.dataset)} images")
    feature_start = time.time()
    feat_bar = SimpleProgressBar(total=len(val_loader), desc="features")
    all_features, all_pids, all_camids, all_paths = pipeline.extract_features(
        val_loader,
        progress_callback=feat_bar.update,
    )
    feat_bar.finish(f"done in {time.time() - feature_start:.1f}s")
    print(f"Feature shape: {all_features.shape}")
    sys.stdout.flush()

    data = split_query_gallery(all_features, all_pids, all_camids, all_paths, num_query)
    q_features = data[0]
    g_features = data[4]
    progress.complete_step(
        1,
        f"query_features={q_features.shape}, gallery_features={g_features.shape}",
    )

    save_path = args.save_path if args.save_path else os.path.join(cfg.OUTPUT_DIR, "cross_modal_results")

    if args.compare_metrics:
        progress.start_step(2, "Compare multiple distance metrics")
        results = run_metric_comparison(args, pipeline, data, save_path)
        progress.complete_step(2, "metric comparison calibration/evaluation done")
        progress.start_step(3, "Comparison metric matrices")
        progress.complete_step(3, "metric comparison matrices done")
        progress.start_step(4, "Comparison metrics")
        progress.complete_step(4, "metric comparison metrics done")
        progress.start_step(5, "Save comparison summary")
        progress.complete_step(5, f"saved to {save_path}")
    else:
        results = run_single_metric(args, pipeline, data, progress, save_path)

    print("\nTesting completed.")
    progress.print_summary()
    return results


if __name__ == "__main__":
    main()
