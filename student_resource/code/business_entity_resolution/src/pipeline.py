"""
Main pipeline for Business Entity Resolution.

Orchestrates: data loading -> preprocessing -> blocking -> feature computation
-> training/inference -> output generation.

Usage (from student_resource/):
    python code/business_entity_resolution/src/pipeline.py --mode full
    python code/business_entity_resolution/src/pipeline.py --mode sample --sample-size 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from blocking import generate_candidates
from features import compute_features_batch, compute_idf_weights
from matching import (
    load_model,
    predict_matches,
    predict_matches_similarity,
    save_model,
    train_matcher,
    tune_similarity_threshold_f05,
)
from preprocessing import preprocess_dataframe


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------


def load_data(data_dir: str, prefix: str = "train") -> tuple:
    """Load source files and (optionally) ground truth."""
    print(f"\n{'=' * 60}")
    print(f"Loading {prefix} data from {data_dir}")
    print(f"{'=' * 60}")

    s1 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source3.tsv"), sep="\t")

    print(f"  Source 1: {len(s1):,} records")
    print(f"  Source 2: {len(s2):,} records")
    print(f"  Source 3: {len(s3):,} records")

    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"  {name} countries: {dict(df['country'].value_counts())}")

    gt = None
    gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, sep="\t")
        print(f"  Ground truth: {len(gt):,} S1 entities")
        n_with = gt["matched_entity_ids"].fillna("").astype(str).str.strip().ne("").sum()
        print(f"    With matches: {n_with:,}")
        print(f"    Singletons: {len(gt) - n_with:,}")

    return s1, s2, s3, gt


def build_ground_truth_lookup(gt: pd.DataFrame) -> dict[str, set[str]]:
    """Convert ground truth DataFrame to {s1_id: set of matched ids}."""
    lookup: dict[str, set[str]] = {}
    for _, row in gt.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or str(matched).strip() == "":
            lookup[s1_id] = set()
        else:
            lookup[s1_id] = {m.strip() for m in str(matched).split(",") if m.strip()}
    return lookup


def sample_training_subset(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    sample_size: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out a reproducible S1 subset plus all referenced S2/S3 records."""
    rng = np.random.RandomState(seed)
    n = min(sample_size, len(s1))
    sampled_ids = set(rng.choice(s1["entity_id"].astype(str).values, size=n, replace=False))

    s1_s = s1[s1["entity_id"].astype(str).isin(sampled_ids)].copy()
    gt_s = gt[gt["source1_entity_id"].astype(str).isin(sampled_ids)].copy()

    needed: set[str] = set()
    for matched in gt_s["matched_entity_ids"].fillna(""):
        if str(matched).strip():
            needed.update(m.strip() for m in str(matched).split(",") if m.strip())

    # Keep true matches plus a random slice of S2/S3 so blocking negatives exist.
    extra_n = min(max(n * 20, 10_000), len(s2) + len(s3))
    s2_ids = s2["entity_id"].astype(str).values
    s3_ids = s3["entity_id"].astype(str).values
    extra = set(rng.choice(np.concatenate([s2_ids, s3_ids]), size=extra_n, replace=False))
    keep = needed | extra

    s2_s = s2[s2["entity_id"].astype(str).isin(keep)].copy()
    s3_s = s3[s3["entity_id"].astype(str).isin(keep)].copy()

    print(
        f"Sampled subset: S1={len(s1_s):,}, S2={len(s2_s):,}, S3={len(s3_s):,}, "
        f"GT={len(gt_s):,}"
    )
    return s1_s, s2_s, s3_s, gt_s


# ---------------------------------------------------------------------------
# Training-pair construction
# ---------------------------------------------------------------------------


def create_training_pairs(
    gt_lookup: dict[str, set[str]],
    candidate_pairs: dict[str, set[str]],
    valid_s2s3: set[str],
    max_neg_ratio: int = 5,
) -> dict[str, list[str]]:
    """Build labelled candidate map for feature computation.

    Returns a candidate_pairs-style dict that includes true matches (even if
    blocking missed them) plus downsampled negatives from the blocker.
    Labels are attached later via ``label_pairs``.
    """
    training_cands: dict[str, list[str]] = {}
    n_pos = n_neg = 0

    for s1_id, candidates in tqdm(candidate_pairs.items(), desc="Building train pairs"):
        if s1_id not in gt_lookup:
            continue

        true_matches = {m for m in gt_lookup[s1_id] if m in valid_s2s3}
        cand_neg = [c for c in candidates if c not in true_matches and c in valid_s2s3]

        if len(cand_neg) > max_neg_ratio * max(len(true_matches), 1):
            rng = np.random.RandomState(abs(hash(s1_id)) % (2**31))
            keep_n = max_neg_ratio * max(len(true_matches), 1)
            cand_neg = list(rng.choice(cand_neg, size=keep_n, replace=False))

        paired = sorted(true_matches) + sorted(cand_neg)
        if paired:
            training_cands[s1_id] = paired
            n_pos += len(true_matches)
            n_neg += len(cand_neg)

    # Ensure S1 entities with true matches but empty blockers still contribute positives.
    for s1_id, true_matches in gt_lookup.items():
        if s1_id in training_cands:
            continue
        kept = sorted(m for m in true_matches if m in valid_s2s3)
        if kept:
            training_cands[s1_id] = kept
            n_pos += len(kept)

    print(f"  Training pairs: {n_pos:,} positive, {n_neg:,} negative")
    return training_cands


def label_feature_rows(
    features_df: pd.DataFrame,
    gt_lookup: dict[str, set[str]],
) -> np.ndarray:
    """Attach binary match labels to a feature frame."""
    labels = np.zeros(len(features_df), dtype=np.int32)
    s1_ids = features_df["s1_id"].astype(str).values
    s2s3_ids = features_df["s2s3_id"].astype(str).values
    for i in range(len(features_df)):
        if s2s3_ids[i] in gt_lookup.get(s1_ids[i], set()):
            labels[i] = 1
    return labels


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def generate_output(
    matches: dict[str, list[str]],
    all_s1_ids: list[str],
    output_path: str,
) -> None:
    """Write matching_results.tsv ensuring every S1 entity has a row."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = sorted(set(matches.get(s1_id, [])))
            f.write(f"{s1_id}\t{','.join(matched)}\n")

    n_with = sum(1 for s1_id in all_s1_ids if matches.get(s1_id))
    print(f"\nOutput written to {output_path}")
    print(f"  Total S1 entities: {len(all_s1_ids):,}")
    print(f"  With matches: {n_with:,}")
    print(f"  Singletons: {len(all_s1_ids) - n_with:,}")


def generate_candidate_output(
    candidates: dict[str, set[str]],
    all_s1_ids: list[str],
    output_path: str,
) -> None:
    """Write candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = sorted(candidates.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    total = sum(len(candidates.get(s1_id, set())) for s1_id in all_s1_ids)
    print(f"Candidates written to {output_path}")
    print(f"  Total candidate pairs: {total:,}")
    print(f"  Avg candidates per S1: {total / max(len(all_s1_ids), 1):.1f}")


def evaluate_on_train(
    matches: dict[str, list[str]],
    gt_lookup: dict[str, set[str]],
) -> float:
    """Macro-averaged F_0.5 over all Source 1 entities."""
    scores = []

    for s1_id, true_matches in gt_lookup.items():
        predicted = set(matches.get(s1_id, []))

        if not true_matches and not predicted:
            scores.append(1.0)
            continue
        if not true_matches and predicted:
            scores.append(0.0)
            continue
        if not predicted and true_matches:
            scores.append(0.0)
            continue

        tp = len(predicted & true_matches)
        fp = len(predicted - true_matches)
        fn = len(true_matches - predicted)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((1.25 * precision * recall) / (0.25 * precision + recall))

    return float(np.mean(scores)) if scores else 0.0


def evaluate_blocking_recall(
    candidate_pairs: dict[str, set[str]],
    gt_lookup: dict[str, set[str]],
) -> float:
    total_true = found_true = 0
    for s1_id, true_matches in gt_lookup.items():
        if not true_matches:
            continue
        candidates = candidate_pairs.get(s1_id, set())
        total_true += len(true_matches)
        found_true += len(true_matches & candidates)
    recall = found_true / max(total_true, 1)
    print(f"\nBlocking recall (upper bound): {recall:.4f}")
    print(f"  Found {found_true:,}/{total_true:,} true matches in candidates")
    return recall


# ---------------------------------------------------------------------------
# Train / test pipelines
# ---------------------------------------------------------------------------


def run_train_pipeline(
    train_dir: str,
    output_dir: str,
    model_path: str,
    sample_size: int | None = None,
    matcher: str = "xgboost",
):
    """Full training pipeline."""
    s1, s2, s3, gt = load_data(train_dir, "train")
    if sample_size is not None:
        s1, s2, s3, gt = sample_training_subset(s1, s2, s3, gt, sample_size)

    gt_lookup = build_ground_truth_lookup(gt)

    print("\n" + "=" * 60)
    print("Preprocessing...")
    print("=" * 60)
    t0 = time.time()
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    print(f"Preprocessing done in {time.time() - t0:.1f}s")

    print("\nComputing IDF weights...")
    all_tokens = (
        s1["name_tokens"].tolist()
        + s2["name_tokens"].tolist()
        + s3["name_tokens"].tolist()
    )
    idf_weights = compute_idf_weights(all_tokens)
    print(f"  Vocabulary size: {len(idf_weights):,}")

    print("\n" + "=" * 60)
    print("Generating candidates (blocking)...")
    print("=" * 60)
    t0 = time.time()
    candidate_pairs = generate_candidates(s1, s2, s3)
    print(f"Blocking done in {time.time() - t0:.1f}s")
    evaluate_blocking_recall(candidate_pairs, gt_lookup)

    print("\n" + "=" * 60)
    print("Creating training data and computing features...")
    print("=" * 60)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    valid_s2s3 = set(s2s3["entity_id"].astype(str))
    train_cands = create_training_pairs(gt_lookup, candidate_pairs, valid_s2s3)

    t0 = time.time()
    features_df = compute_features_batch(
        s1, s2s3, train_cands, idf_weights=idf_weights, n_workers=1
    )
    print(f"Feature computation done in {time.time() - t0:.1f}s")

    if features_df.empty:
        print("ERROR: No training data generated. Check blocking recall.")
        return None

    labels = label_feature_rows(features_df, gt_lookup)
    s1_ids = features_df["s1_id"].astype(str).values
    feature_cols = [c for c in features_df.columns if c not in ("s1_id", "s2s3_id")]

    print("\n" + "=" * 60)
    if matcher == "xgboost":
        from matching import HAS_XGBOOST

        if not HAS_XGBOOST:
            print("XGBoost unavailable — falling back to similarity matcher")
            matcher = "similarity"

    if matcher == "xgboost":
        print("Training XGBoost matcher...")
        print("=" * 60)
        model, threshold, feature_cols = train_matcher(
            features_df, labels, feature_cols, s1_ids
        )
        save_model(model, threshold, feature_cols, model_path)
        use_xgb = True
    else:
        print("Tuning similarity-threshold matcher...")
        print("=" * 60)
        threshold = tune_similarity_threshold_f05(features_df, labels, s1_ids)
        model = {"matcher": "similarity"}
        save_model(model, threshold, feature_cols, model_path)
        use_xgb = False

    print("\n" + "=" * 60)
    print("Evaluating on training candidates...")
    print("=" * 60)

    # Score every blocked candidate (not just the subsampled train pairs).
    eval_cands = {
        sid: sorted(cands)
        for sid, cands in candidate_pairs.items()
        if cands
    }
    t0 = time.time()
    eval_features = compute_features_batch(
        s1, s2s3, eval_cands, idf_weights=idf_weights, n_workers=1
    )
    print(f"Eval feature computation done in {time.time() - t0:.1f}s")

    if use_xgb and model is not None and not eval_features.empty:
        matches = predict_matches(
            model,
            eval_features,
            feature_cols,
            threshold,
            eval_features["s1_id"].astype(str).values,
            eval_features["s2s3_id"].astype(str).values,
        )
    elif not eval_features.empty:
        matches = predict_matches_similarity(eval_features, threshold=threshold)
    else:
        matches = {}

    train_f05 = evaluate_on_train(matches, gt_lookup)
    print(f"\n  Training F_0.5: {train_f05:.4f}")

    # Persist candidates/matches for the sample run as a sanity check.
    all_s1_ids = s1["entity_id"].astype(str).tolist()
    generate_output(
        matches, all_s1_ids, os.path.join(output_dir, "train_matching_results.tsv")
    )
    generate_candidate_output(
        candidate_pairs, all_s1_ids, os.path.join(output_dir, "train_candidate_pairs.tsv")
    )

    print("\nTraining pipeline complete.")
    return model, threshold, feature_cols, idf_weights, use_xgb


def run_test_pipeline(
    test_dir: str,
    output_dir: str,
    model_path: str,
    idf_weights=None,
    use_xgb: bool = True,
):
    """Run inference on test data and write submission files."""
    s1, s2, s3, _ = load_data(test_dir, "test")

    print("\n" + "=" * 60)
    print("Preprocessing test data...")
    print("=" * 60)
    t0 = time.time()
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    print(f"Preprocessing done in {time.time() - t0:.1f}s")

    if idf_weights is None:
        print("\nComputing IDF weights from test data...")
        all_tokens = (
            s1["name_tokens"].tolist()
            + s2["name_tokens"].tolist()
            + s3["name_tokens"].tolist()
        )
        idf_weights = compute_idf_weights(all_tokens)

    model, threshold, feature_cols = load_model(model_path)
    if isinstance(model, dict) and model.get("matcher") == "similarity":
        use_xgb = False
    print(f"\nLoaded model with threshold={threshold:.3f} (xgboost={use_xgb})")

    print("\n" + "=" * 60)
    print("Generating candidates for test data...")
    print("=" * 60)
    t0 = time.time()
    candidate_pairs = generate_candidates(s1, s2, s3)
    print(f"Blocking done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("Computing features and predicting...")
    print("=" * 60)

    s2s3 = pd.concat([s2, s3], ignore_index=True)
    eval_cands = {sid: sorted(cands) for sid, cands in candidate_pairs.items() if cands}
    print(f"  S1 entities with candidates: {len(eval_cands):,}")
    print(
        f"  Total pairs to score: "
        f"{sum(len(v) for v in eval_cands.values()):,}"
    )

    if eval_cands:
        t0 = time.time()
        eval_features = compute_features_batch(
            s1, s2s3, eval_cands, idf_weights=idf_weights, n_workers=1
        )
        print(f"Feature computation done in {time.time() - t0:.1f}s")

        if use_xgb and not isinstance(model, dict):
            matches = predict_matches(
                model,
                eval_features,
                feature_cols,
                threshold,
                eval_features["s1_id"].astype(str).values,
                eval_features["s2s3_id"].astype(str).values,
            )
        else:
            matches = predict_matches_similarity(eval_features, threshold=threshold)
    else:
        matches = {}

    print("\n" + "=" * 60)
    print("Generating output files...")
    print("=" * 60)

    all_s1_ids = s1["entity_id"].astype(str).tolist()
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    generate_output(matches, all_s1_ids, matching_path)
    generate_candidate_output(candidate_pairs, all_s1_ids, candidate_path)

    print("\nTest pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "full", "sample"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-path", default="output/model.pkl")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="S1 entities to use in --mode sample",
    )
    parser.add_argument(
        "--matcher",
        choices=["xgboost", "similarity"],
        default="xgboost",
        help="Matching strategy",
    )
    args = parser.parse_args()

    # Resolve paths relative to student_resource/ when launched from anywhere.
    repo_root = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
    if not os.path.isabs(args.train_dir):
        args.train_dir = os.path.join(repo_root, args.train_dir)
    if not os.path.isabs(args.test_dir):
        args.test_dir = os.path.join(repo_root, args.test_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(repo_root, args.output_dir)
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(repo_root, args.model_path)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "sample":
        result = run_train_pipeline(
            args.train_dir,
            args.output_dir,
            args.model_path,
            sample_size=args.sample_size,
            matcher=args.matcher,
        )
        if result is None:
            sys.exit(1)
        return

    if args.mode in ("train", "full"):
        result = run_train_pipeline(
            args.train_dir,
            args.output_dir,
            args.model_path,
            matcher=args.matcher,
        )
        if args.mode == "full" and result is not None:
            _, _, _, idf_weights, use_xgb = result
            run_test_pipeline(
                args.test_dir,
                args.output_dir,
                args.model_path,
                idf_weights=idf_weights,
                use_xgb=use_xgb,
            )

    elif args.mode == "test":
        run_test_pipeline(args.test_dir, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()
