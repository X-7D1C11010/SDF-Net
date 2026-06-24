import numpy as np

from utils.distance_metrics import compute_distance, metric_type


class ThresholdClassifier:
    """Threshold-based binary matcher with supervised and unsupervised calibration."""

    def __init__(
        self,
        threshold=None,
        metric_type="distance",
        threshold_strategy="mad",
        threshold_percentile=95.0,
        threshold_mad_scale=3.0,
    ):
        self.threshold = threshold
        self.metric_type = metric_type
        self.threshold_strategy = threshold_strategy
        self.threshold_percentile = threshold_percentile
        self.threshold_mad_scale = threshold_mad_scale
        self.calibration_stats = {}

    @property
    def lower_is_match(self):
        return self.metric_type == "distance"

    def fit(self, X, y=None):
        if y is None:
            self.calibrate_unsupervised(X)
        else:
            self.find_optimal_threshold(np.asarray(X).reshape(-1), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        if self.threshold is None:
            raise RuntimeError("Threshold has not been calibrated.")
        X = np.asarray(X)
        if self.lower_is_match:
            return (X <= self.threshold).astype(np.int32)
        return (X >= self.threshold).astype(np.int32)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.lower_is_match:
            scores = 1.0 - self._minmax(X)
        else:
            scores = self._minmax(X)
        return np.stack([1.0 - scores, scores], axis=-1)

    def set_threshold(self, threshold):
        self.threshold = float(threshold)

    def calibrate_unsupervised(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)

        best_values = values.min(axis=1) if self.lower_is_match else values.max(axis=1)
        best_values = best_values[np.isfinite(best_values)]
        if best_values.size == 0:
            raise ValueError("Cannot calibrate threshold from empty or non-finite values.")

        if self.threshold is not None and self.threshold_strategy == "manual":
            threshold = float(self.threshold)
        elif self.threshold_strategy == "percentile":
            threshold = self._percentile_threshold(best_values)
        elif self.threshold_strategy == "otsu":
            threshold = self._otsu_threshold(best_values)
        elif self.threshold_strategy == "mad":
            threshold = self._mad_threshold(best_values)
        else:
            raise ValueError(
                "Unknown threshold strategy: "
                f"{self.threshold_strategy}. Use manual, mad, percentile, or otsu."
            )

        self.threshold = float(threshold)
        self.calibration_stats = {
            "strategy": self.threshold_strategy,
            "metric_type": self.metric_type,
            "threshold": self.threshold,
            "best_min": float(np.min(best_values)),
            "best_max": float(np.max(best_values)),
            "best_mean": float(np.mean(best_values)),
            "best_median": float(np.median(best_values)),
            "best_std": float(np.std(best_values)),
            "best_count": int(best_values.size),
        }
        return self.threshold

    def find_optimal_threshold(self, X, y):
        X = np.asarray(X, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.int32).reshape(-1)
        finite = np.isfinite(X)
        X = X[finite]
        y = y[finite]
        if X.size == 0:
            raise ValueError("Cannot find threshold from empty values.")

        thresholds = np.linspace(np.min(X), np.max(X), 200)
        best_threshold = float(thresholds[0])
        best_f1 = -1.0
        best_accuracy = 0.0

        for threshold in thresholds:
            self.threshold = float(threshold)
            pred = self.predict(X)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = np.mean(pred == y)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
                best_accuracy = float(accuracy)

        self.threshold = best_threshold
        self.calibration_stats = {
            "strategy": "supervised_f1",
            "threshold": self.threshold,
            "best_f1": float(best_f1),
            "best_accuracy": best_accuracy,
        }
        return best_threshold, best_accuracy

    def _mad_threshold(self, best_values):
        median = np.median(best_values)
        mad = np.median(np.abs(best_values - median))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 1e-12:
            return self._percentile_threshold(best_values)
        if self.lower_is_match:
            return median + self.threshold_mad_scale * robust_sigma
        return median - self.threshold_mad_scale * robust_sigma

    def _percentile_threshold(self, best_values):
        pct = float(self.threshold_percentile)
        pct = min(max(pct, 0.0), 100.0)
        if self.lower_is_match:
            return np.percentile(best_values, pct)
        return np.percentile(best_values, 100.0 - pct)

    @staticmethod
    def _otsu_threshold(best_values, bins=128):
        hist, bin_edges = np.histogram(best_values, bins=bins)
        if hist.sum() == 0:
            return float(np.median(best_values))

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        weights_1 = np.cumsum(hist)
        weights_2 = np.cumsum(hist[::-1])[::-1]
        mean_1 = np.cumsum(hist * centers) / np.maximum(weights_1, 1)
        mean_2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weights_2[::-1], 1))[::-1]
        variance = weights_1[:-1] * weights_2[1:] * (mean_1[:-1] - mean_2[1:]) ** 2
        if variance.size == 0:
            return float(np.median(best_values))
        return float(centers[:-1][np.argmax(variance)])

    @staticmethod
    def _minmax(values):
        v_min = np.nanmin(values)
        v_max = np.nanmax(values)
        if abs(v_max - v_min) <= 1e-12:
            return np.ones_like(values, dtype=np.float32)
        return (values - v_min) / (v_max - v_min)


class DistanceBasedClassifier:
    """Optional supervised classifier for pairwise distances."""

    def __init__(self, classifier_type="svm", **kwargs):
        self.classifier_type = classifier_type
        self.classifier = None
        self.scaler = None
        self._initialize_classifier(**kwargs)

    def _initialize_classifier(self, **kwargs):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        self.scaler = StandardScaler()
        if self.classifier_type == "svm":
            self.classifier = SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                gamma=kwargs.get("gamma", "scale"),
                probability=True,
                class_weight="balanced",
            )
        elif self.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=kwargs.get("C", 1.0),
                penalty=kwargs.get("penalty", "l2"),
                solver=kwargs.get("solver", "lbfgs"),
                max_iter=kwargs.get("max_iter", 1000),
                class_weight="balanced",
            )
        elif self.classifier_type == "knn":
            self.classifier = KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5),
                weights=kwargs.get("weights", "uniform"),
                metric=kwargs.get("metric", "euclidean"),
            )
        elif self.classifier_type == "decision_tree":
            self.classifier = DecisionTreeClassifier(
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                class_weight="balanced",
            )
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                class_weight="balanced",
                random_state=42,
            )
        elif self.classifier_type == "gradient_boosting":
            self.classifier = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 3),
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(np.asarray(X).reshape(-1, 1))
        return self.classifier.predict(X_scaled).reshape(np.asarray(X).shape)

    def predict_proba(self, X):
        shape = np.asarray(X).shape
        X_scaled = self.scaler.transform(np.asarray(X).reshape(-1, 1))
        return self.classifier.predict_proba(X_scaled).reshape(*shape, -1)


class CrossModalMatcher:
    """Cross-modal matcher that can operate without test-label calibration."""

    def __init__(
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
        self.distance_metric = distance_metric
        self.classifier_type = classifier_type
        self.threshold_strategy = "manual" if threshold is not None else threshold_strategy
        self.threshold_percentile = threshold_percentile
        self.threshold_mad_scale = threshold_mad_scale
        self.require_mutual = require_mutual
        self.classifier = None
        self._last_values = None
        self._initialize_classifier(threshold=threshold, **kwargs)

    def _initialize_classifier(self, threshold=None, **kwargs):
        if self.classifier_type == "threshold":
            self.classifier = ThresholdClassifier(
                threshold=threshold,
                metric_type=metric_type(self.distance_metric),
                threshold_strategy=self.threshold_strategy,
                threshold_percentile=self.threshold_percentile,
                threshold_mad_scale=self.threshold_mad_scale,
            )
        else:
            self.classifier = DistanceBasedClassifier(
                classifier_type=self.classifier_type,
                **kwargs,
            )

    def fit(self, q_features, g_features, q_pids=None, g_pids=None, supervised=False):
        values = self._compute_values(q_features, g_features)
        self._last_values = values

        if self.classifier_type == "threshold" and not supervised:
            threshold = self.classifier.calibrate_unsupervised(values)
            return {"threshold": threshold, "num_pairs": int(values.size)}

        if q_pids is None or g_pids is None:
            raise ValueError("Supervised matcher fitting requires q_pids and g_pids.")

        q_pids = np.asarray(q_pids)
        g_pids = np.asarray(g_pids)
        y = (q_pids[:, None] == g_pids[None, :]).astype(np.int32).reshape(-1)
        X = values.reshape(-1, 1)

        if self.classifier_type == "threshold":
            threshold, accuracy = self.classifier.find_optimal_threshold(X[:, 0], y)
            return {"threshold": threshold, "accuracy": accuracy, "num_pairs": int(values.size)}

        sample_size = min(50000, len(y))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(y), sample_size, replace=False)
        self.classifier.fit(X[indices], y[indices])
        return {"num_pairs": int(values.size), "sample_size": int(sample_size)}

    def predict(self, q_features, g_features):
        values = self._get_or_compute_values(q_features, g_features)
        pred = self.classifier.predict(values)
        if self.require_mutual:
            pred = pred * self._mutual_top1_mask(values)
        return pred

    def predict_proba(self, q_features, g_features):
        values = self._get_or_compute_values(q_features, g_features)
        return self.classifier.predict_proba(values)

    def _compute_values(self, q_features, g_features):
        return compute_distance(q_features, g_features, metric=self.distance_metric)

    def _get_or_compute_values(self, q_features, g_features):
        if (
            self._last_values is not None
            and q_features.shape[0] == self._last_values.shape[0]
            and g_features.shape[0] == self._last_values.shape[1]
        ):
            return self._last_values
        self._last_values = self._compute_values(q_features, g_features)
        return self._last_values

    def _mutual_top1_mask(self, values):
        if metric_type(self.distance_metric) == "distance":
            q_best = np.argmin(values, axis=1)
            g_best = np.argmin(values, axis=0)
        else:
            q_best = np.argmax(values, axis=1)
            g_best = np.argmax(values, axis=0)

        mask = np.zeros_like(values, dtype=np.int32)
        for q_idx, g_idx in enumerate(q_best):
            if g_best[g_idx] == q_idx:
                mask[q_idx, g_idx] = 1
        return mask

    def get_params(self):
        params = {
            "distance_metric": self.distance_metric,
            "classifier_type": self.classifier_type,
            "require_mutual": self.require_mutual,
        }
        if hasattr(self.classifier, "threshold"):
            params.update(
                {
                    "threshold": self.classifier.threshold,
                    "threshold_strategy": self.classifier.threshold_strategy,
                    "threshold_percentile": self.classifier.threshold_percentile,
                    "threshold_mad_scale": self.classifier.threshold_mad_scale,
                    "calibration_stats": self.classifier.calibration_stats,
                }
            )
        return params


__classifiers__ = {
    "threshold": ThresholdClassifier,
    "svm": DistanceBasedClassifier,
    "logistic_regression": DistanceBasedClassifier,
    "knn": DistanceBasedClassifier,
    "decision_tree": DistanceBasedClassifier,
    "random_forest": DistanceBasedClassifier,
    "gradient_boosting": DistanceBasedClassifier,
}


def create_classifier(classifier_type="threshold", **kwargs):
    if classifier_type == "threshold":
        return ThresholdClassifier(**kwargs)
    return DistanceBasedClassifier(classifier_type=classifier_type, **kwargs)
