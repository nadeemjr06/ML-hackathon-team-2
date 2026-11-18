"""Model registry: >= 5 candidate classifiers from different algorithm families.

Every model is a binary classifier P(candidate is the true arc). Models that
are expensive in training/inference (KNN, MLP) are fitted on a random subset of
*observations* (all candidates of a sampled observation are kept together);
the cap is part of the recorded configuration.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

# name -> (family, factory, params, max_train_rows or None)
REGISTRY = {
    "logreg": ("Linear", lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p)),
               dict(C=1.0, max_iter=2000), None),
    "gaussian_nb": ("Probabilistic", lambda p: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED), GaussianNB(**p)),
                    dict(var_smoothing=1e-9), None),
    "knn": ("Instance-based", lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
            dict(n_neighbors=50, weights="distance", n_jobs=-1), 150_000),
    "decision_tree": ("Tree-based", lambda p: DecisionTreeClassifier(**p),
                      dict(max_depth=12, min_samples_leaf=50, random_state=SEED), None),
    "random_forest": ("Ensemble (bagging)", lambda p: RandomForestClassifier(**p),
                      dict(n_estimators=150, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED), None),
    "lightgbm": ("Boosting", lambda p: LGBMClassifier(**p),
                 dict(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1), None),
    "mlp": ("Neural network", lambda p: make_pipeline(StandardScaler(), MLPClassifier(**p)),
            dict(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                 max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                 random_state=SEED), 400_000),
}


def build(name: str, params: dict | None = None):
    family, factory, default, cap = REGISTRY[name]
    p = dict(default)
    if params:
        p.update(params)
    return factory(p), family, p, cap
