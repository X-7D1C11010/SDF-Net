import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ThresholdClassifier:
    def __init__(self, threshold=0.5, metric_type="distance"):
        self.threshold = threshold
        self.metric_type = metric_type

    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.metric_type == "distance":
            return (X < self.threshold).astype(int)
        else:
            return (X >= self.threshold).astype(int)

    def predict_proba(self, X):
        if self.metric_type == "distance":
            prob = 1.0 - X / (np.max(X) + 1e-5)
        else:
            prob = X / (np.max(X) + 1e-5)
        return np.column_stack([1 - prob, prob])

    def set_threshold(self, threshold):
        self.threshold = threshold

    def find_optimal_threshold(self, X, y):
        thresholds = np.linspace(np.min(X), np.max(X), 100)
        best_threshold = self.threshold
        best_f1 = 0.0
        best_accuracy = 0.0

        for t in thresholds:
            self.threshold = t
            predictions = self.predict(X)
            
            tp = np.sum(predictions * y)
            fp = np.sum(predictions * (1 - y))
            fn = np.sum((1 - predictions) * y)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = np.mean(predictions == y)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_accuracy = accuracy

        self.threshold = best_threshold
        return best_threshold, best_accuracy


class DistanceBasedClassifier:
    def __init__(self, classifier_type="svm", **kwargs):
        self.classifier_type = classifier_type
        self.classifier = None
        self.scaler = StandardScaler()
        self._initialize_classifier(**kwargs)

    def _initialize_classifier(self, **kwargs):
        if self.classifier_type == "svm":
            self.classifier = SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                gamma=kwargs.get("gamma", "scale"),
                probability=True,
                class_weight="balanced"
            )
        elif self.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=kwargs.get("C", 1.0),
                penalty=kwargs.get("penalty", "l2"),
                solver=kwargs.get("solver", "lbfgs"),
                max_iter=kwargs.get("max_iter", 1000),
                class_weight="balanced"
            )
        elif self.classifier_type == "knn":
            self.classifier = KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5),
                weights=kwargs.get("weights", "uniform"),
                metric=kwargs.get("metric", "euclidean")
            )
        elif self.classifier_type == "decision_tree":
            self.classifier = DecisionTreeClassifier(
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                class_weight="balanced"
            )
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                class_weight="balanced",
                random_state=42
            )
        elif self.classifier_type == "gradient_boosting":
            self.classifier = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 3),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def score(self, X, y):
        X_scaled = self.scaler.transform(X)
        return self.classifier.score(X_scaled, y)

    def tune_hyperparameters(self, X, y, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.classifier)
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='f1')
        grid_search.fit(X, y)

        self.classifier = grid_search.best_estimator_.named_steps['classifier']
        self.scaler = grid_search.best_estimator_.named_steps['scaler']

        return grid_search.best_params_, grid_search.best_score_

    def _get_default_param_grid(self):
        if self.classifier_type == "svm":
            return {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__kernel': ['linear', 'rbf', 'poly']
            }
        elif self.classifier_type == "logistic_regression":
            return {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                'classifier__solver': ['lbfgs', 'saga']
            }
        elif self.classifier_type == "knn":
            return {
                'classifier__n_neighbors': [3, 5, 7, 9, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
        elif self.classifier_type == "random_forest":
            return {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10]
            }
        else:
            return {}


class CrossModalMatcher:
    def __init__(self, distance_metric="euclidean", classifier_type="threshold", **kwargs):
        self.distance_metric = distance_metric
        self.classifier_type = classifier_type
        self.classifier = None
        self._initialize_classifier(**kwargs)

    def _initialize_classifier(self, **kwargs):
        if self.classifier_type == "threshold":
            metric_type = "distance" if self.distance_metric in ["euclidean", "manhattan", "chebyshev", "minkowski", "mahalanobis", "hybrid"] else "similarity"
            self.classifier = ThresholdClassifier(
                threshold=kwargs.get("threshold", 0.5),
                metric_type=metric_type
            )
        else:
            self.classifier = DistanceBasedClassifier(
                classifier_type=self.classifier_type,
                **kwargs
            )

    def fit(self, q_features, g_features, q_pids, g_pids):
        import sys
        import time
        
        fit_start = time.time()
        print(f"        → Computing distance matrix for training...")
        sys.stdout.flush()
        distances = self._compute_distances(q_features, g_features)
        print(f"        ✓ Distance matrix shape: {distances.shape} ({time.time()-fit_start:.2f}s)")
        sys.stdout.flush()

        q_pids_arr = np.asarray(q_pids)
        g_pids_arr = np.asarray(g_pids)
        
        q_pids_expanded = q_pids_arr[:, np.newaxis]
        g_pids_expanded = g_pids_arr[np.newaxis, :]
        
        y_matrix = (q_pids_expanded == g_pids_expanded).astype(np.int32)
        
        X = distances.flatten().reshape(-1, 1)
        y = y_matrix.flatten()
        print(f"        → Training data: {X.shape[0]} samples, {np.sum(y)} positive ({np.sum(y)/len(y)*100:.2f}%)")
        sys.stdout.flush()

        if self.classifier_type == "threshold":
            print(f"        → Finding optimal threshold...")
            sys.stdout.flush()
            self.classifier.find_optimal_threshold(X[:, 0], y)
            print(f"        ✓ Optimal threshold: {self.classifier.threshold:.4f}")
            sys.stdout.flush()
        else:
            sample_size = min(50000, len(y))
            print(f"        → Training {self.classifier_type} with {sample_size} samples...")
            sys.stdout.flush()
            np.random.seed(42)
            indices = np.random.choice(len(y), sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
            self.classifier.fit(X_sample, y_sample)
            print(f"        ✓ Classifier trained ({time.time()-fit_start:.2f}s)")
            sys.stdout.flush()

        self._last_distances = distances
        return X, y

    def predict(self, q_features, g_features):
        if hasattr(self, '_last_distances') and self._last_distances is not None and \
           q_features.shape[0] == self._last_distances.shape[0] and g_features.shape[0] == self._last_distances.shape[1]:
            distances = self._last_distances
        else:
            distances = self._compute_distances(q_features, g_features)
            self._last_distances = distances

        if self.classifier_type == "threshold":
            return self.classifier.predict(distances)
        else:
            return self.classifier.predict(distances)

    def predict_proba(self, q_features, g_features):
        if hasattr(self, '_last_distances') and self._last_distances is not None and \
           q_features.shape[0] == self._last_distances.shape[0] and g_features.shape[0] == self._last_distances.shape[1]:
            distances = self._last_distances
        else:
            distances = self._compute_distances(q_features, g_features)
            self._last_distances = distances

        if self.classifier_type == "threshold":
            return self.classifier.predict_proba(distances)
        else:
            return self.classifier.predict_proba(distances)

    def _compute_distances(self, q_features, g_features):
        from utils.distance_metrics import compute_distance
        return compute_distance(q_features, g_features, metric=self.distance_metric)

    def get_params(self):
        if hasattr(self.classifier, 'threshold'):
            return {"distance_metric": self.distance_metric, "threshold": self.classifier.threshold}
        else:
            return {"distance_metric": self.distance_metric, "classifier_type": self.classifier_type}


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
    else:
        return DistanceBasedClassifier(classifier_type=classifier_type, **kwargs)