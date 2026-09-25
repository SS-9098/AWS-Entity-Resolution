"""Pairwise similarity feature computation for entity resolution.

Computes a comprehensive set of string-similarity, phonetic, address,
and metadata features for candidate entity pairs.  Each pair consists
of a Source 1 record and a Source 2 / Source 3 record.

Dependencies:
    - rapidfuzz   (fast string distances)
    - jellyfish   (soundex / metaphone)
    - numpy, pandas, tqdm
"""

from __future__ import annotations

import math
import multiprocessing as mp
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import Indel, Levenshtein as RapidLevenshtein
from tqdm import tqdm

__all__ = [
    # Name features
    "name_levenshtein",
    "name_affine_gap",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_overlap_coeff",
    "name_token_idf_cosine",
    # Phonetic features
    "soundex_match_ratio",
    "metaphone_match_ratio",
    # Address features
    "address_levenshtein",
    "address_token_overlap",
    "city_match",
    "state_match",
    "pincode_match",
    "address_composite_score",
    # Meta features
    "country_match",
    "name_length_ratio",
    "common_token_count",
    "name_vs_address_weighted_score",
    # Batch / orchestration
    "compute_features",
    "compute_features_batch",
    "compute_idf_weights",
]

# ---------------------------------------------------------------------------
# Name features
# ---------------------------------------------------------------------------


def name_levenshtein(name1: str, name2: str) -> float:
    """Normalised Levenshtein similarity (0–1, 1 = identical).

    Uses *rapidfuzz* for speed.  The score is defined as
    ``1 - (edit_distance / max(len(name1), len(name2)))``.
    """
    if not name1 or not name2:
        return 0.0
    sim = RapidLevenshtein.normalized_similarity(name1, name2)
    return float(sim)


def name_affine_gap(name1: str, name2: str) -> float:
    """Affine-gap (indel) similarity for names (0–1).

    Uses rapidfuzz's Indel distance, which applies a lower marginal cost to
    consecutive insertions/deletions than Levenshtein. That helps with
    abbreviation and token-length mismatches (e.g. ``corp`` vs ``corporation``).
    """
    if not name1 or not name2:
        return 0.0
    return float(Indel.normalized_similarity(name1, name2))


def name_jaro_winkler(name1: str, name2: str) -> float:
    """Jaro-Winkler similarity (0–1) via *jellyfish*."""
    if not name1 or not name2:
        return 0.0
    return float(jellyfish.jaro_winkler_similarity(name1, name2))


def name_token_sort_ratio(name1: str, name2: str) -> float:
    """Token-sort ratio from *rapidfuzz* (handles word reordering).

    Returns a value in [0, 1].
    """
    if not name1 or not name2:
        return 0.0
    return float(rfuzz.token_sort_ratio(name1, name2) / 100.0)


def name_token_set_ratio(name1: str, name2: str) -> float:
    """Token-set ratio from *rapidfuzz* (handles subset names).

    Returns a value in [0, 1].
    """
    if not name1 or not name2:
        return 0.0
    return float(rfuzz.token_set_ratio(name1, name2) / 100.0)


def name_partial_ratio(name1: str, name2: str) -> float:
    """Partial ratio from *rapidfuzz* (handles substring matching).

    Returns a value in [0, 1].
    """
    if not name1 or not name2:
        return 0.0
    return float(rfuzz.partial_ratio(name1, name2) / 100.0)


def name_jaccard(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Jaccard similarity of two token sets.

    ``|A ∩ B| / |A ∪ B|``
    """
    set1, set2 = set(tokens1 or []), set(tokens2 or [])
    if not set1 or not set2:
        return 0.0
    return float(len(set1 & set2) / len(set1 | set2))


def name_overlap_coeff(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Overlap coefficient: ``|A ∩ B| / min(|A|, |B|)``."""
    set1, set2 = set(tokens1 or []), set(tokens2 or [])
    if not set1 or not set2:
        return 0.0
    return float(len(set1 & set2) / min(len(set1), len(set2)))


def name_token_idf_cosine(
    tokens1: Sequence[str],
    tokens2: Sequence[str],
    idf_weights: Optional[Dict[str, float]] = None,
) -> float:
    """TF-IDF weighted cosine similarity of two token sets.

    Each token's weight is its IDF value (defaulting to 1.0 for
    unknown tokens).  The TF component is binary (present / absent).
    """
    set1, set2 = set(tokens1 or []), set(tokens2 or [])
    if not set1 or not set2:
        return 0.0
    if idf_weights is None:
        idf_weights = {}

    all_tokens = set1 | set2
    default_idf = 1.0

    vec1 = np.array(
        [idf_weights.get(t, default_idf) if t in set1 else 0.0 for t in all_tokens]
    )
    vec2 = np.array(
        [idf_weights.get(t, default_idf) if t in set2 else 0.0 for t in all_tokens]
    )

    dot = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Phonetic features
# ---------------------------------------------------------------------------


def _soundex_safe(token: str) -> Optional[str]:
    """Return the soundex code for *token*, or ``None`` on failure."""
    try:
        return jellyfish.soundex(token)
    except Exception:
        return None


def _metaphone_safe(token: str) -> Optional[str]:
    """Return the metaphone code for *token*, or ``None`` on failure."""
    try:
        return jellyfish.metaphone(token)
    except Exception:
        return None


def soundex_match_ratio(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Fraction of tokens in *tokens1* whose soundex code matches any in *tokens2*."""
    list1 = list(tokens1 or [])
    list2 = list(tokens2 or [])
    if not list1 or not list2:
        return 0.0

    codes2 = {_soundex_safe(t) for t in list2} - {None}
    if not codes2:
        return 0.0

    matches = sum(1 for t in list1 if _soundex_safe(t) in codes2)
    return float(matches / len(list1))


def metaphone_match_ratio(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Fraction of tokens in *tokens1* whose metaphone code matches any in *tokens2*."""
    list1 = list(tokens1 or [])
    list2 = list(tokens2 or [])
    if not list1 or not list2:
        return 0.0

    codes2 = {_metaphone_safe(t) for t in list2} - {None}
    if not codes2:
        return 0.0

    matches = sum(1 for t in list1 if _metaphone_safe(t) in codes2)
    return float(matches / len(list1))


# ---------------------------------------------------------------------------
# Address features
# ---------------------------------------------------------------------------


def address_levenshtein(addr1: str, addr2: str) -> float:
    """Normalised Levenshtein similarity of full normalised addresses."""
    if not addr1 or not addr2:
        return 0.0
    return float(RapidLevenshtein.normalized_similarity(addr1, addr2))


def address_token_overlap(addr1: str, addr2: str) -> float:
    """Token-level Jaccard similarity of address strings."""
    if not addr1 or not addr2:
        return 0.0
    set1 = set(addr1.split())
    set2 = set(addr2.split())
    if not set1 or not set2:
        return 0.0
    return float(len(set1 & set2) / len(set1 | set2))


def city_match(
    comp1: Optional[Dict[str, str]], comp2: Optional[Dict[str, str]]
) -> float:
    """1.0 if the cities match (fuzzy, threshold ≥ 85), else 0.0."""
    c1 = (comp1 or {}).get("city", "") or ""
    c2 = (comp2 or {}).get("city", "") or ""
    if not c1 or not c2:
        return 0.0
    ratio = rfuzz.ratio(c1.lower(), c2.lower())
    return 1.0 if ratio >= 85 else 0.0


def state_match(
    comp1: Optional[Dict[str, str]], comp2: Optional[Dict[str, str]]
) -> float:
    """1.0 if the states match exactly (case-insensitive), else 0.0."""
    s1 = (comp1 or {}).get("state", "") or ""
    s2 = (comp2 or {}).get("state", "") or ""
    if not s1 or not s2:
        return 0.0
    return 1.0 if s1.lower() == s2.lower() else 0.0


def pincode_match(
    comp1: Optional[Dict[str, str]], comp2: Optional[Dict[str, str]]
) -> float:
    """1.0 if pin/zip codes match (stripped), else 0.0."""
    p1 = (comp1 or {}).get("pin_code", "") or ""
    p2 = (comp2 or {}).get("pin_code", "") or ""
    p1, p2 = p1.strip(), p2.strip()
    if not p1 or not p2:
        return 0.0
    return 1.0 if p1 == p2 else 0.0


def address_composite_score(
    addr1: str,
    addr2: str,
    comp1: Optional[Dict[str, str]],
    comp2: Optional[Dict[str, str]],
) -> float:
    """Address similarity as a sum of raw components (then normalised).

    ``score = (raw_levenshtein + token_overlap + city + state + pin) / 5``

    Each term is in [0, 1], so the result is also in [0, 1]. Keeping the
    additive form (rather than a hand-tuned weighted average) lets the
    downstream matcher learn relative importance.
    """
    return float(
        (
            address_levenshtein(addr1, addr2)
            + address_token_overlap(addr1, addr2)
            + city_match(comp1, comp2)
            + state_match(comp1, comp2)
            + pincode_match(comp1, comp2)
        )
        / 5.0
    )


def name_vs_address_weighted_score(
    name_feats: Dict[str, float],
    address_score: float,
    name_weight: float = 0.70,
    addr_weight: float = 0.30,
) -> float:
    """Name-heavy weighted blend of name and address similarity.

    Default weights put more emphasis on name than address, matching the
    precision-heavy F_0.5 objective (false name merges are costly).
    """
    name_keys = (
        "name_levenshtein",
        "name_affine_gap",
        "name_jaro_winkler",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_jaccard",
        "name_overlap_coeff",
    )
    vals = [name_feats[k] for k in name_keys if k in name_feats]
    name_score = float(sum(vals) / len(vals)) if vals else 0.0
    return float(name_weight * name_score + addr_weight * address_score)


# ---------------------------------------------------------------------------
# Meta features
# ---------------------------------------------------------------------------


def country_match(country1: str, country2: str) -> float:
    """1.0 if the two country values match (case-insensitive), else 0.0."""
    c1 = (country1 or "").strip().lower()
    c2 = (country2 or "").strip().lower()
    if not c1 or not c2:
        return 0.0
    return 1.0 if c1 == c2 else 0.0


def name_length_ratio(name1: str, name2: str) -> float:
    """``min(len, len) / max(len, len)`` of two name strings."""
    l1, l2 = len(name1 or ""), len(name2 or "")
    if l1 == 0 or l2 == 0:
        return 0.0
    return float(min(l1, l2) / max(l1, l2))


def common_token_count(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Integer count of shared tokens (returned as float for consistency)."""
    set1, set2 = set(tokens1 or []), set(tokens2 or [])
    return float(len(set1 & set2))


# ---------------------------------------------------------------------------
# IDF weights
# ---------------------------------------------------------------------------


def compute_idf_weights(all_name_tokens: List[List[str]]) -> Dict[str, float]:
    """Compute IDF weights from all entities' name-token lists.

    ``IDF(t) = log(N / df(t))`` where *df(t)* is the number of entities
    whose token list contains *t*, and *N* is the total entity count.

    Parameters
    ----------
    all_name_tokens:
        A list where each element is one entity's list of name tokens.

    Returns
    -------
    dict mapping each token to its IDF weight.
    """
    n = len(all_name_tokens)
    if n == 0:
        return {}

    doc_freq: Dict[str, int] = {}
    for token_list in all_name_tokens:
        for token in set(token_list or []):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    return {token: math.log(n / df) for token, df in doc_freq.items()}


# ---------------------------------------------------------------------------
# Single-pair feature computation
# ---------------------------------------------------------------------------

def _safe_get(record: Any, key: str, default: Any = "") -> Any:
    """Retrieve *key* from a dict-like or pandas Series, returning *default*
    if the key is missing or the value is null / NaN."""
    try:
        val = record[key] if isinstance(record, dict) else record.get(key, default)  # type: ignore[union-attr]
    except (KeyError, TypeError):
        return default
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val


def _to_str(val: Any) -> str:
    """Coerce *val* to ``str``, treating None/NaN as empty."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val)


def _to_list(val: Any) -> List[str]:
    """Coerce *val* to a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return val.split() if val else []
    return []


def _to_dict(val: Any) -> Dict[str, str]:
    """Coerce *val* to a dict (for addr_components)."""
    if isinstance(val, dict):
        return val
    return {}


def compute_features(
    s1_record: Union[dict, pd.Series],
    s2s3_record: Union[dict, pd.Series],
    idf_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute all pairwise features for one (Source 1, Source 2/3) pair.

    Parameters
    ----------
    s1_record, s2s3_record:
        Dict-like objects with keys: ``entity_id``, ``business_name``,
        ``business_address``, ``country``, ``norm_name``, ``norm_address``,
        ``name_tokens``, ``addr_components``.
    idf_weights:
        Optional pre-computed IDF weights (see :func:`compute_idf_weights`).

    Returns
    -------
    dict[str, float]
        Flat dictionary mapping feature names to their float values.
    """
    # -- extract fields --
    name1 = _to_str(_safe_get(s1_record, "norm_name"))
    name2 = _to_str(_safe_get(s2s3_record, "norm_name"))

    tokens1 = _to_list(_safe_get(s1_record, "name_tokens", []))
    tokens2 = _to_list(_safe_get(s2s3_record, "name_tokens", []))

    addr1 = _to_str(_safe_get(s1_record, "norm_address"))
    addr2 = _to_str(_safe_get(s2s3_record, "norm_address"))

    comp1 = _to_dict(_safe_get(s1_record, "addr_components", {}))
    comp2 = _to_dict(_safe_get(s2s3_record, "addr_components", {}))

    country1 = _to_str(_safe_get(s1_record, "country"))
    country2 = _to_str(_safe_get(s2s3_record, "country"))

    name_lev = name_levenshtein(name1, name2)
    name_aff = name_affine_gap(name1, name2)
    name_jw = name_jaro_winkler(name1, name2)
    name_tsort = name_token_sort_ratio(name1, name2)
    name_tset = name_token_set_ratio(name1, name2)
    name_partial = name_partial_ratio(name1, name2)
    name_jac = name_jaccard(tokens1, tokens2)
    name_ov = name_overlap_coeff(tokens1, tokens2)
    name_idf = name_token_idf_cosine(tokens1, tokens2, idf_weights)
    sx = soundex_match_ratio(tokens1, tokens2)
    mp = metaphone_match_ratio(tokens1, tokens2)
    addr_lev = address_levenshtein(addr1, addr2)
    addr_tok = address_token_overlap(addr1, addr2)
    city = city_match(comp1, comp2)
    state = state_match(comp1, comp2)
    pin = pincode_match(comp1, comp2)
    addr_comp = address_composite_score(addr1, addr2, comp1, comp2)

    feats = {
        # Name
        "name_levenshtein": name_lev,
        "name_affine_gap": name_aff,
        "name_jaro_winkler": name_jw,
        "name_token_sort_ratio": name_tsort,
        "name_token_set_ratio": name_tset,
        "name_partial_ratio": name_partial,
        "name_jaccard": name_jac,
        "name_overlap_coeff": name_ov,
        "name_token_idf_cosine": name_idf,
        # Phonetic
        "soundex_match_ratio": sx,
        "metaphone_match_ratio": mp,
        # Address
        "address_levenshtein": addr_lev,
        "address_token_overlap": addr_tok,
        "city_match": city,
        "state_match": state,
        "pincode_match": pin,
        "address_composite_score": addr_comp,
        # Meta
        "country_match": country_match(country1, country2),
        "name_length_ratio": name_length_ratio(name1, name2),
        "common_token_count": common_token_count(tokens1, tokens2),
    }
    feats["name_address_weighted"] = name_vs_address_weighted_score(feats, addr_comp)
    return feats


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

# Column ordering used by the batch function.
_FEATURE_COLUMNS: List[str] = [
    "name_levenshtein",
    "name_affine_gap",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_overlap_coeff",
    "name_token_idf_cosine",
    "soundex_match_ratio",
    "metaphone_match_ratio",
    "address_levenshtein",
    "address_token_overlap",
    "city_match",
    "state_match",
    "pincode_match",
    "address_composite_score",
    "country_match",
    "name_length_ratio",
    "common_token_count",
    "name_address_weighted",
]


def _compute_pair_features(
    pair: tuple,
    s1_lookup: Dict[str, dict],
    s2s3_lookup: Dict[str, dict],
    idf_weights: Optional[Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    """Compute features for a single pair (worker function).

    Returns a dict with ``s1_id``, ``s2s3_id``, and all feature values,
    or ``None`` if either record is missing.
    """
    s1_id, s2s3_id = pair
    s1_rec = s1_lookup.get(str(s1_id))
    s2s3_rec = s2s3_lookup.get(str(s2s3_id))
    if s1_rec is None or s2s3_rec is None:
        return None
    feats = compute_features(s1_rec, s2s3_rec, idf_weights)
    feats["s1_id"] = s1_id
    feats["s2s3_id"] = s2s3_id
    return feats


def _process_chunk(
    chunk: List[tuple],
    s1_lookup: Dict[str, dict],
    s2s3_lookup: Dict[str, dict],
    idf_weights: Optional[Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Process a chunk of pairs – used by the worker pool."""
    results: List[Dict[str, Any]] = []
    for pair in chunk:
        row = _compute_pair_features(pair, s1_lookup, s2s3_lookup, idf_weights)
        if row is not None:
            results.append(row)
    return results


def _df_to_lookup(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert a DataFrame into a ``{entity_id: row_dict}`` lookup.

    Only retain the columns needed for feature computation to keep the
    in-memory lookup as small as possible.
    """
    if "entity_id" not in df.columns:
        # Already indexed by entity_id
        work = df.copy()
        work = work.reset_index()
        if "entity_id" not in work.columns and work.columns[0] != "entity_id":
            # First column after reset is likely the old index name
            first = work.columns[0]
            work = work.rename(columns={first: "entity_id"})
    else:
        work = df

    keep_cols = [
        c
        for c in (
            "entity_id",
            "norm_name",
            "norm_address",
            "name_tokens",
            "addr_components",
            "country",
        )
        if c in work.columns
    ]
    work = work[keep_cols]
    records = work.to_dict(orient="records")
    return {str(rec["entity_id"]): rec for rec in records}


def compute_features_batch(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    candidate_pairs: Mapping[str, Iterable[str]],
    idf_weights: Optional[Dict[str, float]] = None,
    chunk_size: int = 5_000,
    n_workers: Optional[int] = 1,
) -> pd.DataFrame:
    """Compute features for all candidate pairs in batch.

    Parameters
    ----------
    s1_df:
        DataFrame of Source 1 records (must contain ``entity_id``).
    s2s3_df:
        DataFrame of Source 2 / Source 3 records (must contain ``entity_id``).
    candidate_pairs:
        Mapping ``{s1_entity_id: [s2s3_entity_id, ...]}`` listing the
        candidate matches for each Source 1 entity.
    idf_weights:
        Optional pre-computed IDF weight dict.
    chunk_size:
        Number of pairs per processing chunk (controls memory).
    n_workers:
        Number of worker processes.  Defaults to ``1`` to minimise memory use.

    Returns
    -------
    pd.DataFrame
        Columns: ``s1_id``, ``s2s3_id``, and all feature columns.
    """
    def _pair_chunks() -> Iterable[List[tuple[str, str]]]:
        chunk: List[tuple[str, str]] = []
        for s1_id, s2s3_ids in candidate_pairs.items():
            for s2s3_id in s2s3_ids:
                chunk.append((str(s1_id), str(s2s3_id)))
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    # -- build lookups --
    s1_lookup = _df_to_lookup(s1_df)
    s2s3_lookup = _df_to_lookup(s2s3_df)

    if n_workers is None:
        n_workers = 1

    result_frames: List[pd.DataFrame] = []

    # Use multiprocessing for large workloads, single-process for small ones.
    if n_workers > 1:
        worker_fn = partial(
            _process_chunk,
            s1_lookup=s1_lookup,
            s2s3_lookup=s2s3_lookup,
            idf_weights=idf_weights,
        )
        with mp.Pool(processes=n_workers) as pool:
            for chunk_result in tqdm(
                pool.imap_unordered(worker_fn, _pair_chunks()),
                desc="Computing features",
                unit="chunk",
            ):
                if chunk_result:
                    result_frames.append(
                        pd.DataFrame.from_records(
                            chunk_result,
                            columns=["s1_id", "s2s3_id"] + _FEATURE_COLUMNS,
                        )
                    )
    else:
        for chunk in tqdm(_pair_chunks(), desc="Computing features", unit="chunk"):
            chunk_rows = _process_chunk(chunk, s1_lookup, s2s3_lookup, idf_weights)
            if chunk_rows:
                result_frames.append(
                    pd.DataFrame.from_records(
                        chunk_rows,
                        columns=["s1_id", "s2s3_id"] + _FEATURE_COLUMNS,
                    )
                )

    if not result_frames:
        return pd.DataFrame(columns=["s1_id", "s2s3_id"] + _FEATURE_COLUMNS)

    if len(result_frames) == 1:
        return result_frames[0]

    return pd.concat(result_frames, ignore_index=True)
