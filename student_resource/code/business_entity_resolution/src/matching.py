"""
Matching / classification module for entity resolution.

Uses XGBoost to classify candidate pairs as match/non-match,
with threshold tuning for F_0.5. Falls back to a name-heavy
similarity score when XGBoost is unavailable.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - environment dependent
    XGBClassifier = None  # type: ignore[misc, assignment]
    _HAS_XGBOOST = False

__all__ = [
    "HAS_XGBOOST",
    "train_matcher",
    "predict_matches",
    "predict_matches_similarity",
    "tune_threshold_f05",
    "tune_similarity_threshold_f05",
    "compute_f05",
    "save_model",
    "load_model",
]

HAS_XGBOOST = _HAS_XGBOOST


def compute_f05(precision: float, recall: float) -> float:
    """Compute F_0.5 score (precision-weighted)."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def compute_f05_per_entity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s1_ids: np.ndarray,
) -> float:
    """
    Compute macro-averaged F_0.5 per Source 1 entity.
    
    This mirrors the official evaluation: F_0.5 is calculated per S1 entity,
    then averaged across all S1 entities.
    """
    unique_s1 = np.unique(s1_ids)
    scores = []
    
    for s1_id in unique_s1:
        mask = s1_ids == s1_id
        true_i = y_true[mask]
        pred_i = y_pred[mask]
        
        tp = np.sum((true_i == 1) & (pred_i == 1))
        fp = np.sum((true_i == 0) & (pred_i == 1))
        fn = np.sum((true_i == 1) & (pred_i == 0))
        
        if tp + fp == 0:
            precision = 1.0 if (tp + fn == 0) else 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)
        
        scores.append(compute_f05(precision, recall))
    
    return np.mean(scores) if scores else 0.0


def train_matcher(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_columns: list[str],
    s1_ids_train: Optional[np.ndarray] = None,
    n_folds: int = 3,
) -> tuple:
    """
    Train an XGBoost classifier for match/non-match prediction.

    Returns
    -------
    (trained_model, best_threshold, feature_columns)
    """
    if not _HAS_XGBOOST:
        raise RuntimeError(
            "XGBoost is not available in this environment. "
            "Install libomp (macOS: brew install libomp) or use --matcher similarity."
        )

    X = X_train[feature_columns].values
    y = y_train

    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"Training XGBoost: {len(y)} samples, {n_pos} positive, {n_neg} negative")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  Features: {len(feature_columns)}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X, y)

    best_threshold = tune_threshold_f05(
        model, X, y, s1_ids_train, feature_columns, n_folds
    )

    print(f"  Best threshold: {best_threshold:.3f}")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  Top 10 features:")
    for i in range(min(10, len(feature_columns))):
        idx = sorted_idx[i]
        print(f"    {feature_columns[idx]}: {importances[idx]:.4f}")

    return model, best_threshold, feature_columns


def tune_threshold_f05(
    model,
    X: np.ndarray,
    y: np.ndarray,
    s1_ids: Optional[np.ndarray],
    feature_columns: list[str],
    n_folds: int = 3,
) -> float:
    """
    Tune the decision threshold to maximize F_0.5 using cross-validation.

    Since F_0.5 is precision-heavy, the optimal threshold is typically > 0.5.
    """
    if not _HAS_XGBOOST:
        raise RuntimeError("XGBoost is required for tune_threshold_f05")

    if s1_ids is None:
        s1_ids = np.arange(len(y))

    thresholds = np.arange(0.20, 0.95, 0.02)
    fold_scores = {t: [] for t in thresholds}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        s1_val = s1_ids[val_idx]

        fold_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        fold_model.fit(X_tr, y_tr)

        proba = fold_model.predict_proba(X_val)[:, 1]

        for t in thresholds:
            preds = (proba >= t).astype(int)
            score = compute_f05_per_entity(y_val, preds, s1_val)
            fold_scores[t].append(score)

    avg_scores = {t: np.mean(scores) for t, scores in fold_scores.items()}
    best_threshold = max(avg_scores, key=avg_scores.get)

    print(f"  CV F_0.5 scores around best threshold:")
    for t in sorted(avg_scores.keys()):
        if abs(t - best_threshold) <= 0.06:
            marker = " <-- best" if t == best_threshold else ""
            print(f"    threshold={t:.2f}: F_0.5={avg_scores[t]:.4f}{marker}")

    return float(best_threshold)


def predict_matches(
    model,
    X: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    s1_ids: np.ndarray,
    s2s3_ids: np.ndarray,
) -> dict[str, list[str]]:
    """
    Predict matches using the trained model and threshold.

    Returns
    -------
    dict[str, list[str]]
        ``{s1_entity_id: [matched s2/s3 entity_ids]}``
    """
    X_feat = X[feature_columns].values
    proba = model.predict_proba(X_feat)[:, 1]

    matches: dict[str, list[str]] = {}
    for i in range(len(proba)):
        s1_id = s1_ids[i]
        s2s3_id = s2s3_ids[i]

        if s1_id not in matches:
            matches[s1_id] = []

        if proba[i] >= threshold:
            matches[s1_id].append(s2s3_id)

    return matches


def predict_matches_similarity(
    features_df: pd.DataFrame,
    name_weight: float = 0.70,
    addr_weight: float = 0.30,
    threshold: float = 0.55,
) -> dict[str, list[str]]:
    """
    Similarity-based matching without an ML model.

    Uses a name-heavy weighted blend of name and address features, then
    applies a threshold tuned for F_0.5 (precision-heavy). Prefer this when
    XGBoost training data is insufficient; otherwise use ``predict_matches``.
    """
    if "name_address_weighted" in features_df.columns:
        composite = features_df["name_address_weighted"]
    else:
        name_cols = [
            "name_levenshtein",
            "name_affine_gap",
            "name_jaro_winkler",
            "name_token_sort_ratio",
            "name_token_set_ratio",
            "name_jaccard",
            "name_overlap_coeff",
        ]
        addr_cols = ["address_composite_score"]

        available_name = [c for c in name_cols if c in features_df.columns]
        available_addr = [c for c in addr_cols if c in features_df.columns]

        name_score = (
            features_df[available_name].mean(axis=1) if available_name else 0.0
        )
        addr_score = (
            features_df[available_addr].mean(axis=1) if available_addr else 0.0
        )
        composite = name_weight * name_score + addr_weight * addr_score

    matches: dict[str, list[str]] = {}
    s1_ids = features_df["s1_id"].values
    s2s3_ids = features_df["s2s3_id"].values
    scores = composite.values if hasattr(composite, "values") else np.asarray(composite)

    for i in range(len(scores)):
        s1_id = s1_ids[i]
        if s1_id not in matches:
            matches[s1_id] = []
        if scores[i] >= threshold:
            matches[s1_id].append(s2s3_ids[i])

    return matches


def tune_similarity_threshold_f05(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    s1_ids: np.ndarray,
    name_weight: float = 0.70,
    addr_weight: float = 0.30,
) -> float:
    """Tune a similarity threshold to maximise macro F_0.5."""
    if "name_address_weighted" in features_df.columns:
        scores = features_df["name_address_weighted"].values
    else:
        name_cols = [
            c
            for c in [
                "name_levenshtein",
                "name_affine_gap",
                "name_jaro_winkler",
                "name_token_sort_ratio",
                "name_token_set_ratio",
                "name_jaccard",
                "name_overlap_coeff",
            ]
            if c in features_df.columns
        ]
        addr_cols = [
            c for c in ["address_composite_score"] if c in features_df.columns
        ]
        name_score = (
            features_df[name_cols].mean(axis=1).values if name_cols else np.zeros(len(features_df))
        )
        addr_score = (
            features_df[addr_cols].mean(axis=1).values if addr_cols else np.zeros(len(features_df))
        )
        scores = name_weight * name_score + addr_weight * addr_score

    best_t, best_score = 0.55, -1.0
    for t in np.arange(0.35, 0.95, 0.02):
        preds = (scores >= t).astype(int)
        score = compute_f05_per_entity(labels, preds, s1_ids)
        if score > best_score:
            best_score = score
            best_t = float(t)

    print(f"  Best similarity threshold: {best_t:.2f} (F_0.5={best_score:.4f})")
    return best_t


def save_model(model, threshold, feature_columns, path):
    """Save trained model, threshold, and feature columns."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "threshold": threshold,
            "feature_columns": feature_columns,
        }, f)
    print(f"Model saved to {path}")


def load_model(path):
    """Load trained model, threshold, and feature columns."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"], data["feature_columns"]
