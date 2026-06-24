import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize


class ComprehensiveMetrics:
    def __init__(self, y_true, y_pred, y_score=None, sample_size=500000):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_score = np.array(y_score) if y_score is not None else None
        self.sample_size = sample_size
        self.metrics = {}
        self._compute_all_metrics()

    def _compute_all_metrics(self):
        total_samples = len(self.y_true)
        if total_samples > self.sample_size:
            np.random.seed(42)
            indices = np.random.choice(total_samples, self.sample_size, replace=False)
            self.y_true_sampled = self.y_true[indices]
            self.y_pred_sampled = self.y_pred[indices]
            self.y_score_sampled = self.y_score[indices] if self.y_score is not None else None
            print(f"      → Sampling {self.sample_size}/{total_samples} for metrics computation...")
            sys.stdout.flush()
        else:
            self.y_true_sampled = self.y_true
            self.y_pred_sampled = self.y_pred
            self.y_score_sampled = self.y_score
        
        self._compute_basic_metrics()
        self._compute_confusion_matrix()
        if self.y_score_sampled is not None:
            self._compute_roc_auc()
            self._compute_precision_recall()

    def _compute_basic_metrics(self):
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_true_sampled, self.y_pred_sampled, average='weighted', zero_division=0)
        self.metrics['recall'] = recall_score(self.y_true_sampled, self.y_pred_sampled, average='weighted', zero_division=0)
        self.metrics['f1'] = f1_score(self.y_true_sampled, self.y_pred_sampled, average='weighted', zero_division=0)

        self.metrics['precision_macro'] = precision_score(self.y_true_sampled, self.y_pred_sampled, average='macro', zero_division=0)
        self.metrics['recall_macro'] = recall_score(self.y_true_sampled, self.y_pred_sampled, average='macro', zero_division=0)
        self.metrics['f1_macro'] = f1_score(self.y_true_sampled, self.y_pred_sampled, average='macro', zero_division=0)

        self.metrics['precision_micro'] = precision_score(self.y_true_sampled, self.y_pred_sampled, average='micro', zero_division=0)
        self.metrics['recall_micro'] = recall_score(self.y_true_sampled, self.y_pred_sampled, average='micro', zero_division=0)
        self.metrics['f1_micro'] = f1_score(self.y_true_sampled, self.y_pred_sampled, average='micro', zero_division=0)

    def _compute_confusion_matrix(self):
        self.cm = confusion_matrix(self.y_true_sampled, self.y_pred_sampled)
        self.metrics['confusion_matrix'] = self.cm.tolist()

        tn, fp, fn, tp = self.cm.ravel() if self.cm.shape == (2, 2) else [0, 0, 0, 0]
        self.metrics['tn'] = int(tn)
        self.metrics['fp'] = int(fp)
        self.metrics['fn'] = int(fn)
        self.metrics['tp'] = int(tp)

        if (tp + tn + fp + fn) > 0:
            self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            self.metrics['balanced_accuracy'] = (self.metrics['sensitivity'] + self.metrics['specificity']) / 2

    def _compute_roc_auc(self):
        if self.y_score_sampled.ndim == 1:
            fpr, tpr, thresholds = roc_curve(self.y_true_sampled, self.y_score_sampled)
            self.metrics['roc_auc'] = auc(fpr, tpr)
            self.metrics['fpr'] = fpr.tolist()
            self.metrics['tpr'] = tpr.tolist()
            self.metrics['thresholds'] = thresholds.tolist()

            optimal_idx = np.argmax(tpr - fpr)
            self.metrics['optimal_threshold'] = float(thresholds[optimal_idx])
        else:
            n_classes = self.y_score_sampled.shape[1]
            y_true_bin = label_binarize(self.y_true_sampled, classes=np.arange(n_classes))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_score_sampled[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), self.y_score_sampled.ravel())
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

            self.metrics['roc_auc_multi'] = roc_auc
            self.metrics['fpr_multi'] = {k: v.tolist() for k, v in fpr.items()}
            self.metrics['tpr_multi'] = {k: v.tolist() for k, v in tpr.items()}

    def _compute_precision_recall(self):
        if self.y_score_sampled.ndim == 1:
            precision, recall, thresholds = precision_recall_curve(self.y_true_sampled, self.y_score_sampled)
            self.metrics['average_precision'] = average_precision_score(self.y_true_sampled, self.y_score_sampled)
            self.metrics['pr_precision'] = precision.tolist()
            self.metrics['pr_recall'] = recall.tolist()
            self.metrics['pr_thresholds'] = thresholds.tolist()
        else:
            n_classes = self.y_score_sampled.shape[1]
            y_true_bin = label_binarize(self.y_true_sampled, classes=np.arange(n_classes))

            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], self.y_score_sampled[:, i])
                average_precision[i] = average_precision_score(y_true_bin[:, i], self.y_score_sampled[:, i])

            precision['micro'], recall['micro'], _ = precision_recall_curve(y_true_bin.ravel(), self.y_score_sampled.ravel())
            average_precision['micro'] = average_precision_score(y_true_bin, self.y_score_sampled, average='micro')

            self.metrics['average_precision_multi'] = average_precision
            self.metrics['pr_precision_multi'] = {k: v.tolist() for k, v in precision.items()}
            self.metrics['pr_recall_multi'] = {k: v.tolist() for k, v in recall.items()}

    def get_metrics(self):
        return self.metrics

    def print_summary(self):
        print("=" * 60)
        print("COMPREHENSIVE METRICS SUMMARY")
        print("=" * 60)

        print("\n--- Basic Metrics ---")
        print(f"Accuracy:          {self.metrics.get('accuracy', 0):.4f}")
        print(f"Precision (weighted): {self.metrics.get('precision', 0):.4f}")
        print(f"Recall (weighted):    {self.metrics.get('recall', 0):.4f}")
        print(f"F1 Score (weighted):  {self.metrics.get('f1', 0):.4f}")
        print(f"Precision (macro):    {self.metrics.get('precision_macro', 0):.4f}")
        print(f"Recall (macro):       {self.metrics.get('recall_macro', 0):.4f}")
        print(f"F1 Score (macro):     {self.metrics.get('f1_macro', 0):.4f}")

        if 'specificity' in self.metrics:
            print("\n--- Binary Classification Metrics ---")
            print(f"Sensitivity (TPR):    {self.metrics.get('sensitivity', 0):.4f}")
            print(f"Specificity (TNR):    {self.metrics.get('specificity', 0):.4f}")
            print(f"Balanced Accuracy:    {self.metrics.get('balanced_accuracy', 0):.4f}")
            print(f"True Positives (TP):  {self.metrics.get('tp', 0)}")
            print(f"True Negatives (TN):  {self.metrics.get('tn', 0)}")
            print(f"False Positives (FP): {self.metrics.get('fp', 0)}")
            print(f"False Negatives (FN): {self.metrics.get('fn', 0)}")

        if 'roc_auc' in self.metrics:
            print("\n--- ROC-AUC ---")
            print(f"AUC Score:            {self.metrics.get('roc_auc', 0):.4f}")
            print(f"Optimal Threshold:    {self.metrics.get('optimal_threshold', 0):.4f}")

        if 'average_precision' in self.metrics:
            print("\n--- Precision-Recall ---")
            print(f"Average Precision:    {self.metrics.get('average_precision', 0):.4f}")

        print("\n--- Confusion Matrix ---")
        if 'confusion_matrix' in self.metrics:
            cm = np.array(self.metrics['confusion_matrix'])
            print(cm)

        print("\n" + "=" * 60)

    def plot_confusion_matrix(self, save_path=None, title='Confusion Matrix'):
        if 'confusion_matrix' not in self.metrics:
            print("Confusion matrix not computed")
            return

        cm = np.array(self.metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        classes = np.unique(self.y_true)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_roc_curve(self, save_path=None, title='ROC Curve'):
        if 'fpr' not in self.metrics or 'tpr' not in self.metrics:
            print("ROC curve not computed")
            return

        fpr = np.array(self.metrics['fpr'])
        tpr = np.array(self.metrics['tpr'])
        roc_auc = self.metrics.get('roc_auc', 0)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_precision_recall_curve(self, save_path=None, title='Precision-Recall Curve'):
        if 'pr_precision' not in self.metrics or 'pr_recall' not in self.metrics:
            print("Precision-Recall curve not computed")
            return

        precision = np.array(self.metrics['pr_precision'])
        recall = np.array(self.metrics['pr_recall'])
        avg_precision = self.metrics.get('average_precision', 0)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()


class ReIDMetrics:
    def __init__(self, distmat, q_pids, g_pids, q_camids=None, g_camids=None, max_rank=50):
        self.distmat = distmat
        self.q_pids = np.array(q_pids)
        self.g_pids = np.array(g_pids)
        self.q_camids = np.array(q_camids) if q_camids is not None else None
        self.g_camids = np.array(g_camids) if g_camids is not None else None
        self.max_rank = max_rank
        self.metrics = {}
        self._compute_all_metrics()

    def _compute_all_metrics(self):
        print(f"      → Computing CMC...")
        sys.stdout.flush()
        cmc_start = time.time()
        self._compute_cmc()
        print(f"      ✓ CMC computed in {time.time()-cmc_start:.2f}s")
        sys.stdout.flush()
        
        print(f"      → Computing mAP...")
        sys.stdout.flush()
        map_start = time.time()
        self._compute_map()
        print(f"      ✓ mAP computed in {time.time()-map_start:.2f}s")
        sys.stdout.flush()
        
        print(f"      → Computing rank metrics...")
        sys.stdout.flush()
        rank_start = time.time()
        self._compute_rank_metrics()
        print(f"      ✓ Rank metrics computed in {time.time()-rank_start:.2f}s")
        sys.stdout.flush()

    def _compute_cmc(self):
        num_q, num_g = self.distmat.shape
        if num_g < self.max_rank:
            self.max_rank = num_g

        use_gpu = torch.cuda.is_available()
        
        if use_gpu and num_q > 100:
            distmat_tensor = torch.tensor(self.distmat, dtype=torch.float32, device='cuda')
            indices = torch.argsort(distmat_tensor, dim=1).cpu().numpy()
            del distmat_tensor
            torch.cuda.empty_cache()
        else:
            indices = np.argsort(self.distmat, axis=1)

        matches = (self.g_pids[indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        for q_idx in range(num_q):
            q_pid = self.q_pids[q_idx]
            order = indices[q_idx]

            if self.q_camids is not None and self.g_camids is not None:
                q_camid = self.q_camids[q_idx]
                remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
                keep = np.invert(remove)
                orig_cmc = matches[q_idx][keep]
            else:
                orig_cmc = matches[q_idx]

            if not np.any(orig_cmc):
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:self.max_rank])

        if len(all_cmc) == 0:
            self.metrics['cmc'] = np.zeros(self.max_rank).tolist()
            return

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / len(all_cmc)
        self.metrics['cmc'] = all_cmc.tolist()

        for r in [1, 5, 10, 20]:
            if r <= self.max_rank:
                self.metrics[f'rank_{r}'] = float(all_cmc[r - 1])

    def _compute_map(self):
        num_q, num_g = self.distmat.shape
        
        use_gpu = torch.cuda.is_available()
        
        if use_gpu and num_q > 100:
            distmat_tensor = torch.tensor(self.distmat, dtype=torch.float32, device='cuda')
            indices = torch.argsort(distmat_tensor, dim=1).cpu().numpy()
            del distmat_tensor
            torch.cuda.empty_cache()
        else:
            indices = np.argsort(self.distmat, axis=1)
            
        matches = (self.g_pids[indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

        all_AP = []
        for q_idx in range(num_q):
            q_pid = self.q_pids[q_idx]
            order = indices[q_idx]

            if self.q_camids is not None and self.g_camids is not None:
                q_camid = self.q_camids[q_idx]
                remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
                keep = np.invert(remove)
                orig_cmc = matches[q_idx][keep]
            else:
                orig_cmc = matches[q_idx]

            if not np.any(orig_cmc):
                continue

            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
            tmp_cmc = tmp_cmc / y
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        if len(all_AP) > 0:
            self.metrics['mAP'] = float(np.mean(all_AP))
            self.metrics['mean_AP'] = float(np.mean(all_AP))
            self.metrics['std_AP'] = float(np.std(all_AP))
        else:
            self.metrics['mAP'] = 0.0
            self.metrics['mean_AP'] = 0.0
            self.metrics['std_AP'] = 0.0

    def _compute_rank_metrics(self):
        num_q = self.distmat.shape[0]
        
        use_gpu = torch.cuda.is_available()
        
        if use_gpu and num_q > 100:
            distmat_tensor = torch.tensor(self.distmat, dtype=torch.float32, device='cuda')
            indices = torch.argsort(distmat_tensor, dim=1).cpu().numpy()
            del distmat_tensor
            torch.cuda.empty_cache()
        else:
            indices = np.argsort(self.distmat, axis=1)
            
        matches = (self.g_pids[indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

        correct_at_rank = {}
        for rank in [1, 5, 10, 20, 50]:
            correct_at_rank[rank] = 0

        for q_idx in range(num_q):
            if self.q_camids is not None and self.g_camids is not None:
                q_pid = self.q_pids[q_idx]
                q_camid = self.q_camids[q_idx]
                order = indices[q_idx]
                remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
                keep = np.invert(remove)
                orig_cmc = matches[q_idx][keep]
            else:
                orig_cmc = matches[q_idx]

            if not np.any(orig_cmc):
                continue

            first_match = np.argmax(orig_cmc) + 1
            for rank in [1, 5, 10, 20, 50]:
                if first_match <= rank:
                    correct_at_rank[rank] += 1

        for rank in [1, 5, 10, 20, 50]:
            if num_q > 0:
                self.metrics[f'accuracy_at_rank_{rank}'] = float(correct_at_rank[rank] / num_q)
            else:
                self.metrics[f'accuracy_at_rank_{rank}'] = 0.0

    def get_metrics(self):
        return self.metrics

    def print_summary(self):
        print("=" * 60)
        print("REID METRICS SUMMARY")
        print("=" * 60)

        print("\n--- mAP ---")
        print(f"mAP:                {self.metrics.get('mAP', 0):.4f}")
        if 'mean_AP' in self.metrics:
            print(f"Mean AP:            {self.metrics.get('mean_AP', 0):.4f}")
            print(f"Std AP:             {self.metrics.get('std_AP', 0):.4f}")

        print("\n--- CMC Rank Metrics ---")
        for r in [1, 5, 10, 20]:
            if f'rank_{r}' in self.metrics:
                print(f"Rank-{r:<3}:          {self.metrics[f'rank_{r}']:.4f}")

        print("\n--- Accuracy at Rank ---")
        for r in [1, 5, 10, 20]:
            if f'accuracy_at_rank_{r}' in self.metrics:
                print(f"Accuracy@Rank-{r:<3}: {self.metrics[f'accuracy_at_rank_{r}']:.4f}")

        print("\n" + "=" * 60)


def compute_matching_metrics(q_features, g_features, q_pids, g_pids, distance_metric="euclidean", classifier_type="threshold"):
    from utils.distance_metrics import compute_distance
    from utils.classifier import CrossModalMatcher

    distmat = compute_distance(q_features, g_features, metric=distance_metric)

    matcher = CrossModalMatcher(
        distance_metric=distance_metric,
        classifier_type=classifier_type
    )

    X, y = matcher.fit(q_features, g_features, q_pids, g_pids)

    y_pred = matcher.predict(q_features, g_features)

    y_true_all = []
    y_pred_all = []
    y_score_all = []

    for i in range(len(q_pids)):
        for j in range(len(g_pids)):
            y_true_all.append(1 if q_pids[i] == g_pids[j] else 0)
            y_pred_all.append(y_pred[i, j])
            if matcher.y_score is not None:
                y_score_all.append(matcher.predict_proba(q_features[i:i+1], g_features[j:j+1])[0, 1])

    y_score_all = np.array(y_score_all) if y_score_all else None

    basic_metrics = ComprehensiveMetrics(y_true_all, y_pred_all, y_score_all)
    reid_metrics = ReIDMetrics(distmat, q_pids, g_pids)

    return {
        'basic': basic_metrics.get_metrics(),
        'reid': reid_metrics.get_metrics(),
        'classifier_params': matcher.get_params()
    }