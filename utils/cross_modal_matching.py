import numpy as np
import os
import json
import sys
import time
from utils.feature_extractor import FeatureExtractor
from utils.distance_metrics import compute_distance
from utils.classifier import CrossModalMatcher
from utils.comprehensive_metrics import ComprehensiveMetrics, ReIDMetrics


class CrossModalMatchingPipeline:
    def __init__(self, cfg, weight_path=None):
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg, weight_path)
        self.matcher = None
        self.results = {}

    def extract_features(self, dataloader, progress_callback=None):
        return self.feature_extractor.extract(dataloader, progress_callback=progress_callback)

    def setup_matcher(self, distance_metric="euclidean", classifier_type="threshold", **kwargs):
        self.matcher = CrossModalMatcher(
            distance_metric=distance_metric,
            classifier_type=classifier_type,
            **kwargs
        )
        return self.matcher

    def train_matcher(self, q_features, g_features, q_pids, g_pids):
        if self.matcher is None:
            self.setup_matcher()

        X, y = self.matcher.fit(q_features, g_features, q_pids, g_pids)
        return X, y

    def match(self, q_features, g_features):
        if self.matcher is None:
            raise ValueError("Matcher not set up. Call setup_matcher() first.")

        predictions = self.matcher.predict(q_features, g_features)
        probs = self.matcher.predict_proba(q_features, g_features)

        return predictions, probs

    def evaluate(self, q_features, g_features, q_pids, g_pids, save_path=None):
        eval_start = time.time()
        print(f"      → Computing distance matrix ({q_features.shape[0]} x {g_features.shape[0]})...")
        sys.stdout.flush()
        distmat = compute_distance(q_features, g_features, metric=self.matcher.distance_metric)
        self._distmat = distmat
        dist_time = time.time() - eval_start
        print(f"      ✓ Distance matrix computed in {dist_time:.2f}s")
        sys.stdout.flush()

        print(f"      → Computing basic classification metrics (direct method)...")
        sys.stdout.flush()
        basic_start = time.time()
        basic_metrics = self._compute_basic_metrics_direct(distmat, q_pids, g_pids)
        basic_time = time.time() - basic_start
        print(f"      ✓ Basic metrics computed in {basic_time:.2f}s")
        sys.stdout.flush()

        print(f"      → Computing ReID metrics (CMC/mAP)...")
        sys.stdout.flush()
        reid_start = time.time()
        reid_metrics = ReIDMetrics(distmat, q_pids, g_pids)
        reid_time = time.time() - reid_start
        print(f"      ✓ ReID metrics computed in {reid_time:.2f}s")
        sys.stdout.flush()

        self.results = {
            'basic': basic_metrics,
            'reid': reid_metrics.get_metrics(),
            'classifier_params': self.matcher.get_params(),
            'distance_metric': self.matcher.distance_metric,
            'classifier_type': self.matcher.classifier_type
        }

        total_eval_time = time.time() - eval_start
        print(f"      ✓ Evaluation completed in {total_eval_time:.2f}s")
        sys.stdout.flush()

        if save_path:
            self.save_results(save_path)

        return self.results

    def _compute_basic_metrics_direct(self, distmat, q_pids, g_pids):
        metrics = {}
        q_pids_arr = np.asarray(q_pids)
        g_pids_arr = np.asarray(g_pids)
        
        threshold = self.matcher.classifier.threshold
        predictions = (distmat < threshold).astype(np.int32)
        
        matches = (q_pids_arr[:, np.newaxis] == g_pids_arr[np.newaxis, :]).astype(np.int32)
        
        tp = np.sum(predictions * matches)
        tn = np.sum((1 - predictions) * (1 - matches))
        fp = np.sum(predictions * (1 - matches))
        fn = np.sum((1 - predictions) * matches)
        
        total = tp + tn + fp + fn
        metrics['accuracy'] = float((tp + tn) / total) if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        metrics['precision_macro'] = metrics['precision']
        metrics['recall_macro'] = metrics['recall']
        metrics['f1_macro'] = metrics['f1']
        metrics['precision_micro'] = metrics['accuracy']
        metrics['recall_micro'] = metrics['accuracy']
        metrics['f1_micro'] = metrics['accuracy']
        
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        
        if (tn + fp) > 0:
            metrics['specificity'] = float(tn / (tn + fp))
        else:
            metrics['specificity'] = 0.0
        metrics['sensitivity'] = float(recall)
        
        if metrics['sensitivity'] + metrics['specificity'] > 0:
            metrics['balanced_accuracy'] = float((metrics['sensitivity'] + metrics['specificity']) / 2)
        else:
            metrics['balanced_accuracy'] = 0.0

        return metrics

    def save_results(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(self.results, f, indent=2)

        if hasattr(self, '_distmat') and self._distmat is not None:
            np.save(os.path.join(save_path, 'distance_matrix.npy'), self._distmat)

        print(f"Results saved to: {save_path}")

    def print_results(self):
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return

        print("=" * 70)
        print("CROSS-MODAL MATCHING RESULTS")
        print("=" * 70)

        print(f"\n--- Configuration ---")
        print(f"Distance Metric:    {self.results.get('distance_metric', 'N/A')}")
        print(f"Classifier Type:    {self.results.get('classifier_type', 'N/A')}")
        print(f"Classifier Params:  {self.results.get('classifier_params', {})}")

        print(f"\n--- Basic Classification Metrics ---")
        basic = self.results.get('basic', {})
        print(f"Accuracy:           {basic.get('accuracy', 0):.4f}")
        print(f"Precision:          {basic.get('precision', 0):.4f}")
        print(f"Recall:             {basic.get('recall', 0):.4f}")
        print(f"F1 Score:           {basic.get('f1', 0):.4f}")

        if 'roc_auc' in basic:
            print(f"\n--- ROC-AUC ---")
            print(f"AUC Score:          {basic.get('roc_auc', 0):.4f}")
            print(f"Optimal Threshold:  {basic.get('optimal_threshold', 0):.4f}")

        print(f"\n--- ReID Metrics ---")
        reid = self.results.get('reid', {})
        print(f"mAP:                {reid.get('mAP', 0):.4f}")
        for r in [1, 5, 10]:
            if f'rank_{r}' in reid:
                print(f"Rank-{r:<3}:          {reid[f'rank_{r}']:.4f}")

        print("\n" + "=" * 70)

    def run_full_pipeline(self, query_loader, gallery_loader, 
                          distance_metric="euclidean", classifier_type="threshold",
                          save_path=None):
        print("=" * 70)
        print("RUNNING FULL CROSS-MODAL MATCHING PIPELINE")
        print("=" * 70)

        print("\nStep 1: Extracting features from query set...")
        q_features, q_pids, q_camids, q_paths = self.extract_features(query_loader)
        print(f"  Query features shape: {q_features.shape}")
        print(f"  Query PIDs: {len(np.unique(q_pids))} unique IDs")

        print("\nStep 2: Extracting features from gallery set...")
        g_features, g_pids, g_camids, g_paths = self.extract_features(gallery_loader)
        print(f"  Gallery features shape: {g_features.shape}")
        print(f"  Gallery PIDs: {len(np.unique(g_pids))} unique IDs")

        print("\nStep 3: Setting up matcher...")
        self.setup_matcher(distance_metric=distance_metric, classifier_type=classifier_type)

        print("\nStep 4: Training matcher...")
        X, y = self.train_matcher(q_features, g_features, q_pids, g_pids)
        print(f"  Training samples: {len(y)}")
        print(f"  Positive samples: {np.sum(y)}")
        print(f"  Negative samples: {len(y) - np.sum(y)}")

        print("\nStep 5: Evaluating...")
        results = self.evaluate(q_features, g_features, q_pids, g_pids, save_path)

        print("\nStep 6: Results summary")
        self.print_results()

        return results


def run_cross_modal_matching(cfg, query_loader, gallery_loader,
                             distance_metric="euclidean", classifier_type="threshold",
                             weight_path=None, save_path=None):
    pipeline = CrossModalMatchingPipeline(cfg, weight_path)
    results = pipeline.run_full_pipeline(
        query_loader, gallery_loader,
        distance_metric=distance_metric,
        classifier_type=classifier_type,
        save_path=save_path
    )
    return results


def compare_distance_metrics(cfg, query_loader, gallery_loader, 
                             metrics=['euclidean', 'cosine_distance', 'manhattan'],
                             classifier_type="threshold", save_path=None, weight_path=None):
    results = {}

    print("Initializing feature extractor...")
    pipeline = CrossModalMatchingPipeline(cfg, weight_path)
    
    print("\nExtracting features from query set...")
    q_features, q_pids, q_camids, q_paths = pipeline.extract_features(query_loader)
    
    print("\nExtracting features from gallery set...")
    g_features, g_pids, g_camids, g_paths = pipeline.extract_features(gallery_loader)

    for metric in metrics:
        print(f"\n{'='*70}")
        print(f"COMPARING: {metric.upper()}")
        print(f"{'='*70}")

        pipeline.setup_matcher(distance_metric=metric, classifier_type=classifier_type)
        X, y = pipeline.train_matcher(q_features, g_features, q_pids, g_pids)
        
        result = pipeline.evaluate(q_features, g_features, q_pids, g_pids, 
                                   os.path.join(save_path, metric) if save_path else None)
        results[metric] = result

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

    return results