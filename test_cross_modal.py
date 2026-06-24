import os
import sys
import argparse
import time
import numpy as np
import json
from tqdm import tqdm
from config import cfg
from datasets import make_dataloader
from utils.cross_modal_matching import CrossModalMatchingPipeline
from utils.progress_bar import MultiStepProgress, SimpleProgressBar


def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Matching Testing")
    parser.add_argument(
        "--config_file",
        default="configs/SDF-Net_Test.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--distance_metric",
        default="euclidean",
        choices=["euclidean", "cosine_distance", "cosine_similarity",
                 "manhattan", "chebyshev", "minkowski", "mahalanobis", "hybrid"],
        help="distance metric to use",
        type=str,
    )
    parser.add_argument(
        "--classifier_type",
        default="threshold",
        choices=["threshold", "svm", "logistic_regression", "knn",
                 "decision_tree", "random_forest", "gradient_boosting"],
        help="classifier type to use",
        type=str,
    )
    parser.add_argument(
        "--compare_metrics",
        action="store_true",
        help="compare multiple distance metrics",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="path to save results",
        type=str,
    )
    parser.add_argument(
        "--weight_path",
        default=None,
        help="path to model weights",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    progress = MultiStepProgress(
        total_steps=6,
        step_names=[
            "加载数据集与模型",
            "提取图像特征向量",
            "训练跨模态匹配器",
            "计算距离矩阵",
            "计算评价指标",
            "保存与输出结果"
        ]
    )

    progress.start_step(0, "初始化 DataLoader 与加载预训练模型权重")
    print("Creating dataloaders...")
    sys.stdout.flush()
    (
        train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
    ) = make_dataloader(cfg, is_train=False)

    print("Dataset statistics:")
    print(f"  Query samples: {num_query}")
    print(f"  Gallery samples: {len(val_loader.dataset) - num_query}")

    model_weight_path = args.weight_path if args.weight_path else cfg.TEST.WEIGHT
    print(f"\nModel weights will be loaded from: {model_weight_path}")
    sys.stdout.flush()

    pipeline = CrossModalMatchingPipeline(cfg, model_weight_path)
    progress.complete_step(0, f"Query: {num_query}, Gallery: {len(val_loader.dataset) - num_query}")

    progress.start_step(1, f"从 {len(val_loader.dataset)} 张图像提取深度特征 (batch_size={cfg.TEST.IMS_PER_BATCH})")
    feature_start = time.time()

    feat_bar = SimpleProgressBar(total=len(val_loader), desc="特征提取")
    all_features, all_pids, all_camids, all_paths = pipeline.extract_features(val_loader, progress_callback=feat_bar.update)
    feat_bar.finish(f"完成 ({time.time()-feature_start:.1f}s)")

    feature_time = time.time() - feature_start
    print(f"  ✓ Feature extraction completed in {feature_time:.2f}s")
    print(f"  ✓ All features shape: {all_features.shape}")
    sys.stdout.flush()

    q_features = all_features[:num_query]
    q_pids = all_pids[:num_query]
    q_camids = all_camids[:num_query]

    g_features = all_features[num_query:]
    g_pids = all_pids[num_query:]
    g_camids = all_camids[num_query:]

    print(f"  ✓ Query features shape: {q_features.shape}")
    print(f"  ✓ Gallery features shape: {g_features.shape}")
    progress.complete_step(1, f"特征维度: {q_features.shape[1]}, 耗时: {feature_time:.1f}s")

    if args.compare_metrics:
        print(f"\n[2/6] Comparing distance metrics...")
        metrics_to_compare = ['euclidean', 'cosine_distance', 'manhattan', 'hybrid']

        save_path = args.save_path if args.save_path else os.path.join(output_dir, 'metric_comparison')

        results = {}
        for metric_idx, metric in enumerate(metrics_to_compare, 1):
            print(f"\n  [{metric_idx}/{len(metrics_to_compare)}] Comparing: {metric.upper()}")
            sys.stdout.flush()

            pipeline.setup_matcher(distance_metric=metric, classifier_type=args.classifier_type)

            print(f"    - Training matcher...")
            sys.stdout.flush()
            X, y = pipeline.train_matcher(q_features, g_features, q_pids, g_pids)
            print(f"    - Training done: {len(y)} samples ({np.sum(y)} positive)")
            sys.stdout.flush()

            print(f"    - Evaluating...")
            sys.stdout.flush()
            result = pipeline.evaluate(q_features, g_features, q_pids, g_pids,
                                       os.path.join(save_path, metric) if save_path else None)
            results[metric] = result
            print(f"    - Evaluation done: mAP={result['reid'].get('mAP', 0):.4f}, Rank-1={result['reid'].get('rank_1', 0):.4f}")
            sys.stdout.flush()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'comparison_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        print(f"\n{'Metric':<20} {'mAP':<10} {'Rank-1':<10} {'Accuracy':<10} {'F1':<10}")
        print(f"{'-----':<20} {'---':<10} {'-----':<10} {'--------':<10} {'--':<10}")

        for metric, result in results.items():
            mAP = result['reid'].get('mAP', 0)
            rank1 = result['reid'].get('rank_1', 0)
            accuracy = result['basic'].get('accuracy', 0)
            f1 = result['basic'].get('f1', 0)

            print(f"{metric:<20} {mAP:<10.4f} {rank1:<10.4f} {accuracy:<10.4f} {f1:<10.4f}")
    else:
        progress.start_step(2, f"训练 {args.classifier_type} 匹配器 (距离度量: {args.distance_metric})")

        save_path = args.save_path if args.save_path else os.path.join(output_dir, 'cross_modal_results')

        print("Setting up matcher...")
        sys.stdout.flush()
        pipeline.setup_matcher(distance_metric=args.distance_metric, classifier_type=args.classifier_type)

        print("Training matcher...")
        sys.stdout.flush()
        train_start = time.time()
        X, y = pipeline.train_matcher(q_features, g_features, q_pids, g_pids)
        train_time = time.time() - train_start

        print(f"  ✓ Training samples: {len(y)}")
        print(f"  ✓ Positive samples: {np.sum(y)}")
        print(f"  ✓ Negative samples: {len(y) - np.sum(y)}")
        print(f"  ✓ Training time: {train_time:.2f}s")
        sys.stdout.flush()

        if pipeline.matcher.classifier_type == "threshold":
            threshold = pipeline.matcher.classifier.threshold
            progress.complete_step(2, f"最优阈值: {threshold:.4f}, 耗时: {train_time:.1f}s")
        else:
            progress.complete_step(2, f"耗时: {train_time:.1f}s")

        progress.start_step(3, f"计算 {q_features.shape[0]}x{g_features.shape[0]} 距离矩阵")
        progress.start_step(4, "计算 mAP/Rank/Accuracy/F1 等评价指标")

        results = pipeline.evaluate(q_features, g_features, q_pids, g_pids, save_path)

        progress.complete_step(3, "距离矩阵计算完成")
        progress.complete_step(4, "评价指标计算完成")

        progress.start_step(5, "保存结果到磁盘并打印摘要")
        pipeline.print_results()
        progress.complete_step(5, f"结果已保存到: {save_path}")

    print("\nTesting completed!")
    progress.print_summary()


if __name__ == "__main__":
    main()