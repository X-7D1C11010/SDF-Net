import csv
import json
import os
import sys
import time

import numpy as np

from utils.classifier import CrossModalMatcher
from utils.distance_metrics import compute_distance, metric_type, to_distance_matrix
from utils.feature_extractor import FeatureExtractor


class CrossModalMatchingPipeline:
    def __init__(self, cfg, weight_path=None):
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg, weight_path)
        self.matcher = None
        self.results = {}
        self._metric_matrix = None
        self._distmat = None
        self._match_matrix = None
        self._top_matches = []

    def extract_features(self, dataloader, progress_callback=None):
        return self.feature_extractor.extract(dataloader, progress_callback=progress_callback)

    def setup_matcher(
        self,
        distance_metric="cosine_distance",
        classifier_type="threshold",
        threshold_strategy="mad",
        threshold_percentile=95.0,
        threshold_mad_scale=3.0,
        threshold=None,
        require_mutual=False,
        **kwargs,
    ):
        self.matcher = CrossModalMatcher(
            distance_metric=distance_metric,
            classifier_type=classifier_type,
            threshold_strategy=threshold_strategy,
            threshold_percentile=threshold_percentile,
            threshold_mad_scale=threshold_mad_scale,
            threshold=threshold,
            require_mutual=require_mutual,
            **kwargs,
        )
        return self.matcher

    def calibrate_matcher(self, q_features, g_features, q_pids=None, g_pids=None, supervised=False):
        if self.matcher is None:
            self.setup_matcher()
        return self.matcher.fit(
            q_features,
            g_features,
            q_pids=q_pids,
            g_pids=g_pids,
            supervised=supervised,
        )

    def train_matcher(self, q_features, g_features, q_pids=None, g_pids=None, supervised=False):
        """Backward-compatible name; defaults to unsupervised calibration."""
        return self.calibrate_matcher(q_features, g_features, q_pids, g_pids, supervised)

    def evaluate(
        self,
        q_features,
        g_features,
        q_pids=None,
        g_pids=None,
        q_camids=None,
        g_camids=None,
        q_paths=None,
        g_paths=None,
        topk=10,
        save_path=None,
    ):
        if self.matcher is None:
            raise ValueError("Matcher is not set up. Call setup_matcher() first.")

        eval_start = time.time()
        print(f"      Computing metric matrix ({q_features.shape[0]} x {g_features.shape[0]})...")
        sys.stdout.flush()
        metric_matrix = compute_distance(q_features, g_features, metric=self.matcher.distance_metric)
        distmat = to_distance_matrix(metric_matrix, self.matcher.distance_metric)
        self.matcher._last_values = metric_matrix
        self._metric_matrix = metric_matrix
        self._distmat = distmat
        print(f"      Metric matrix computed in {time.time() - eval_start:.2f}s")
        sys.stdout.flush()

        match_start = time.time()
        match_matrix = self.matcher.predict(q_features, g_features)
        self._match_matrix = match_matrix.astype(np.int32)
        matching_summary = self._compute_matching_summary(match_matrix, metric_matrix)
        print(f"      Threshold matching computed in {time.time() - match_start:.2f}s")
        sys.stdout.flush()

        basic_metrics = None
        reid_metrics = None
        if q_pids is not None and g_pids is not None:
            basic_metrics = self._compute_basic_metrics(match_matrix, q_pids, g_pids)
            reid_metrics = self._compute_reid_metrics(
                distmat,
                q_pids,
                g_pids,
                q_camids=q_camids,
                g_camids=g_camids,
            )

        self._top_matches = self._build_top_matches(
            metric_matrix,
            distmat,
            match_matrix,
            q_pids=q_pids,
            g_pids=g_pids,
            q_paths=q_paths,
            g_paths=g_paths,
            topk=topk,
        )

        self.results = {
            "matching": matching_summary,
            "basic": basic_metrics or {},
            "reid": reid_metrics or {},
            "classifier_params": self.matcher.get_params(),
            "distance_metric": self.matcher.distance_metric,
            "metric_type": metric_type(self.matcher.distance_metric),
            "classifier_type": self.matcher.classifier_type,
            "topk": int(topk),
        }

        print(f"      Evaluation completed in {time.time() - eval_start:.2f}s")
        sys.stdout.flush()

        if save_path:
            self.save_results(save_path)

        return self.results

    def _compute_matching_summary(self, match_matrix, metric_matrix):
        num_q, num_g = match_matrix.shape
        per_query = match_matrix.sum(axis=1)
        per_gallery = match_matrix.sum(axis=0)
        if metric_type(self.matcher.distance_metric) == "distance":
            best_values = metric_matrix.min(axis=1)
        else:
            best_values = metric_matrix.max(axis=1)

        return {
            "num_queries": int(num_q),
            "num_gallery": int(num_g),
            "accepted_pairs": int(match_matrix.sum()),
            "queries_with_match": int(np.sum(per_query > 0)),
            "gallery_with_match": int(np.sum(per_gallery > 0)),
            "avg_matches_per_query": float(np.mean(per_query)) if num_q > 0 else 0.0,
            "median_best_metric": float(np.median(best_values)) if best_values.size else 0.0,
            "mean_best_metric": float(np.mean(best_values)) if best_values.size else 0.0,
        }

    @staticmethod
    def _compute_basic_metrics(match_matrix, q_pids, g_pids):
        q_pids = np.asarray(q_pids)
        g_pids = np.asarray(g_pids)
        gt = (q_pids[:, None] == g_pids[None, :]).astype(np.int32)
        pred = match_matrix.astype(np.int32)

        tp = int(np.sum((pred == 1) & (gt == 1)))
        tn = int(np.sum((pred == 0) & (gt == 0)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        fn = int(np.sum((pred == 0) & (gt == 1)))
        total = tp + tn + fp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float((tp + tn) / total) if total > 0 else 0.0,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "sensitivity": float(recall),
            "balanced_accuracy": float((recall + specificity) / 2.0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    @staticmethod
    def _compute_reid_metrics(distmat, q_pids, g_pids, q_camids=None, g_camids=None, max_rank=50):
        q_pids = np.asarray(q_pids)
        g_pids = np.asarray(g_pids)
        q_camids = np.asarray(q_camids) if q_camids is not None else None
        g_camids = np.asarray(g_camids) if g_camids is not None else None

        num_q, num_g = distmat.shape
        max_rank = min(max_rank, num_g)
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)

        all_cmc = []
        all_ap = []
        first_match_ranks = []

        for q_idx in range(num_q):
            order = indices[q_idx]
            keep = np.ones(num_g, dtype=bool)
            if q_camids is not None and g_camids is not None:
                keep = ~((g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx]))

            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            cmc = cmc[:max_rank]
            if len(cmc) < max_rank:
                cmc = np.pad(cmc, (0, max_rank - len(cmc)), constant_values=cmc[-1])
            all_cmc.append(cmc)

            num_rel = orig_cmc.sum()
            precision_at_k = orig_cmc.cumsum() / (np.arange(len(orig_cmc)) + 1.0)
            all_ap.append(float((precision_at_k * orig_cmc).sum() / num_rel))
            first_match_ranks.append(int(np.argmax(orig_cmc) + 1))

        if not all_cmc:
            metrics = {
                "mAP": 0.0,
                "valid_queries": 0,
                "cmc": [0.0] * max_rank,
            }
        else:
            cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / len(all_cmc)
            metrics = {
                "mAP": float(np.mean(all_ap)),
                "mean_AP": float(np.mean(all_ap)),
                "std_AP": float(np.std(all_ap)),
                "valid_queries": int(len(all_cmc)),
                "cmc": cmc.tolist(),
                "mean_first_match_rank": float(np.mean(first_match_ranks)),
            }

        for rank in [1, 5, 10, 20, 50]:
            if rank <= max_rank and metrics["cmc"]:
                metrics[f"rank_{rank}"] = float(metrics["cmc"][rank - 1])
                if first_match_ranks:
                    metrics[f"accuracy_at_rank_{rank}"] = float(
                        np.mean(np.asarray(first_match_ranks) <= rank)
                    )
        return metrics

    def _build_top_matches(
        self,
        metric_matrix,
        distmat,
        match_matrix,
        q_pids=None,
        g_pids=None,
        q_paths=None,
        g_paths=None,
        topk=10,
    ):
        topk = min(int(topk), distmat.shape[1])
        order = np.argsort(distmat, axis=1)[:, :topk]
        q_pids = np.asarray(q_pids) if q_pids is not None else None
        g_pids = np.asarray(g_pids) if g_pids is not None else None
        rows = []

        for q_idx in range(order.shape[0]):
            for rank_idx, g_idx in enumerate(order[q_idx], start=1):
                row = {
                    "query_index": int(q_idx),
                    "gallery_index": int(g_idx),
                    "rank": int(rank_idx),
                    "metric_value": float(metric_matrix[q_idx, g_idx]),
                    "ranking_distance": float(distmat[q_idx, g_idx]),
                    "accepted_by_threshold": int(match_matrix[q_idx, g_idx]),
                }
                if q_paths is not None:
                    row["query_path"] = q_paths[q_idx]
                if g_paths is not None:
                    row["gallery_path"] = g_paths[g_idx]
                if q_pids is not None and g_pids is not None:
                    row["query_pid"] = int(q_pids[q_idx])
                    row["gallery_pid"] = int(g_pids[g_idx])
                    row["is_ground_truth_match"] = int(q_pids[q_idx] == g_pids[g_idx])
                rows.append(row)
        return rows

    def save_results(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        if self._metric_matrix is not None:
            np.save(os.path.join(save_path, "metric_matrix.npy"), self._metric_matrix)
        if self._distmat is not None:
            np.save(os.path.join(save_path, "distance_matrix.npy"), self._distmat)
        if self._match_matrix is not None:
            np.save(os.path.join(save_path, "match_matrix.npy"), self._match_matrix)
        if self._top_matches:
            csv_path = os.path.join(save_path, "top_matches.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._top_matches[0].keys()))
                writer.writeheader()
                writer.writerows(self._top_matches)

        print(f"Results saved to: {save_path}")

    def print_results(self):
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return

        print("=" * 70)
        print("CROSS-MODAL MATCHING RESULTS")
        print("=" * 70)
        print(f"Distance metric:   {self.results.get('distance_metric', 'N/A')}")
        print(f"Metric type:       {self.results.get('metric_type', 'N/A')}")
        print(f"Classifier:        {self.results.get('classifier_type', 'N/A')}")
        print(f"Params:            {self.results.get('classifier_params', {})}")

        matching = self.results.get("matching", {})
        print("\n--- Threshold Matching ---")
        print(f"Accepted pairs:    {matching.get('accepted_pairs', 0)}")
        print(f"Queries matched:   {matching.get('queries_with_match', 0)}/{matching.get('num_queries', 0)}")
        print(f"Avg/query:         {matching.get('avg_matches_per_query', 0):.4f}")

        basic = self.results.get("basic", {})
        if basic:
            print("\n--- Binary Pair Metrics ---")
            print(f"Accuracy:          {basic.get('accuracy', 0):.4f}")
            print(f"Precision:         {basic.get('precision', 0):.4f}")
            print(f"Recall:            {basic.get('recall', 0):.4f}")
            print(f"F1:                {basic.get('f1', 0):.4f}")

        reid = self.results.get("reid", {})
        if reid:
            print("\n--- ReID Ranking Metrics ---")
            print(f"mAP:               {reid.get('mAP', 0):.4f}")
            for r in [1, 5, 10, 20]:
                if f"rank_{r}" in reid:
                    print(f"Rank-{r:<3}:         {reid[f'rank_{r}']:.4f}")

        print("=" * 70)

    def run_full_pipeline(
        self,
        query_loader,
        gallery_loader,
        distance_metric="cosine_distance",
        classifier_type="threshold",
        save_path=None,
        **matcher_kwargs,
    ):
        print("=" * 70)
        print("RUNNING CROSS-MODAL MATCHING PIPELINE")
        print("=" * 70)

        print("\nStep 1: Extracting query features...")
        q_features, q_pids, q_camids, q_paths = self.extract_features(query_loader)
        print("\nStep 2: Extracting gallery features...")
        g_features, g_pids, g_camids, g_paths = self.extract_features(gallery_loader)

        print("\nStep 3: Calibrating matcher without labels...")
        self.setup_matcher(
            distance_metric=distance_metric,
            classifier_type=classifier_type,
            **matcher_kwargs,
        )
        self.calibrate_matcher(q_features, g_features)

        print("\nStep 4: Evaluating...")
        results = self.evaluate(
            q_features,
            g_features,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            q_paths=q_paths,
            g_paths=g_paths,
            save_path=save_path,
        )
        self.print_results()
        return results


def run_cross_modal_matching(
    cfg,
    query_loader,
    gallery_loader,
    distance_metric="cosine_distance",
    classifier_type="threshold",
    weight_path=None,
    save_path=None,
    **matcher_kwargs,
):
    pipeline = CrossModalMatchingPipeline(cfg, weight_path)
    return pipeline.run_full_pipeline(
        query_loader,
        gallery_loader,
        distance_metric=distance_metric,
        classifier_type=classifier_type,
        save_path=save_path,
        **matcher_kwargs,
    )


def compare_distance_metrics(
    cfg,
    query_loader,
    gallery_loader,
    metrics=None,
    classifier_type="threshold",
    save_path=None,
    weight_path=None,
    **matcher_kwargs,
):
    if metrics is None:
        metrics = ["cosine_distance", "euclidean", "hybrid", "manhattan"]

    pipeline = CrossModalMatchingPipeline(cfg, weight_path)
    print("\nExtracting query features...")
    q_features, q_pids, q_camids, q_paths = pipeline.extract_features(query_loader)
    print("\nExtracting gallery features...")
    g_features, g_pids, g_camids, g_paths = pipeline.extract_features(gallery_loader)

    results = {}
    for metric in metrics:
        print(f"\n{'=' * 70}")
        print(f"COMPARING: {metric.upper()}")
        print(f"{'=' * 70}")
        pipeline.setup_matcher(
            distance_metric=metric,
            classifier_type=classifier_type,
            **matcher_kwargs,
        )
        pipeline.calibrate_matcher(q_features, g_features)
        result = pipeline.evaluate(
            q_features,
            g_features,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            q_paths=q_paths,
            g_paths=g_paths,
            save_path=os.path.join(save_path, metric) if save_path else None,
        )
        results[metric] = result

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "comparison_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} {'mAP':<10} {'Rank-1':<10} {'Precision':<10} {'F1':<10}")
    for metric, result in results.items():
        print(
            f"{metric:<20} "
            f"{result['reid'].get('mAP', 0):<10.4f} "
            f"{result['reid'].get('rank_1', 0):<10.4f} "
            f"{result['basic'].get('precision', 0):<10.4f} "
            f"{result['basic'].get('f1', 0):<10.4f}"
        )

    return results
