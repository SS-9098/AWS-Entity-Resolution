"""
Blocking strategies for entity resolution candidate generation.

This module generates candidate pairs by matching Source 1 entities against
Source 2 and Source 3 entities. Multiple blocking strategies are combined
(unioned) to maximise recall while keeping the candidate set tractable.

Designed for ~2 M rows per source — every hot path is built around inverted
indices and batch processing to stay memory- and CPU-efficient.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Sequence

import jellyfish
import pandas as pd
from tqdm import tqdm

__all__ = [
    "PredicateBlocker",
    "TokenBlocker",
    "generate_candidates",
    "prune_candidates_by_name",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

def first_n_chars(name: str, n: int = 5) -> str | None:
    """Return the first *n* characters of *name* (already normalised).

    Returns ``None`` when the name is too short to be useful (< 2 chars).
    """
    if not name or len(name) < 2:
        return None
    return name[:n]


def sorted_first_n_tokens(name: str, n: int = 2) -> str | None:
    """Sort the first *n* whitespace-delimited tokens and join them.

    Sorting makes the key order-invariant, which helps when the same
    business name appears with tokens swapped (e.g. "acme corp" vs
    "corp acme").  Returns ``None`` when fewer than 1 token is present.
    """
    tokens = name.split()
    if not tokens:
        return None
    selected = sorted(tokens[:n])
    return " ".join(selected)


def soundex_first_token(name: str) -> str | None:
    """Soundex encoding of the first meaningful token (length >= 2).

    Soundex collapses phonetically similar names ("smith" / "smyth")
    into the same bucket.  Returns ``None`` when no suitable token exists.
    """
    for tok in name.split():
        if len(tok) >= 2:
            try:
                return jellyfish.soundex(tok)
            except Exception:
                return None
    return None


def country_plus_first_chars(name: str, country: str, n: int = 4) -> str | None:
    """Composite key: country code concatenated with first *n* chars of name.

    Restricts candidates to the same country *and* a name prefix match,
    giving a tighter block than either predicate alone.
    """
    if not name or len(name) < 2 or not country:
        return None
    return f"{country}|{name[:n]}"


# ---------------------------------------------------------------------------
# Default predicate factory
# ---------------------------------------------------------------------------

def _build_default_predicates() -> list[Callable[..., str | None]]:
    """Return the standard set of predicate functions.

    Each callable has the signature ``(name: str, country: str) -> str | None``
    so that :class:`PredicateBlocker` can invoke them uniformly.

    Loose keys such as bare 3-char prefixes are avoided on their own; they
    are paired with country to keep block sizes tractable at multi-million
    row scale.
    """

    def _first5(name: str, country: str) -> str | None:          # noqa: ARG001
        return first_n_chars(name, n=5)

    def _sorted2(name: str, country: str) -> str | None:         # noqa: ARG001
        return sorted_first_n_tokens(name, n=2)

    def _sorted3(name: str, country: str) -> str | None:         # noqa: ARG001
        return sorted_first_n_tokens(name, n=3)

    def _soundex(name: str, country: str) -> str | None:         # noqa: ARG001
        return soundex_first_token(name)

    def _country4(name: str, country: str) -> str | None:
        return country_plus_first_chars(name, country, n=4)

    def _country5(name: str, country: str) -> str | None:
        return country_plus_first_chars(name, country, n=5)

    _first5.__qualname__ = "first_n_chars(n=5)"
    _sorted2.__qualname__ = "sorted_first_n_tokens(n=2)"
    _sorted3.__qualname__ = "sorted_first_n_tokens(n=3)"
    _soundex.__qualname__ = "soundex_first_token"
    _country4.__qualname__ = "country_plus_first_chars(n=4)"
    _country5.__qualname__ = "country_plus_first_chars(n=5)"

    return [_first5, _sorted2, _sorted3, _soundex, _country4, _country5]


# ---------------------------------------------------------------------------
# PredicateBlocker
# ---------------------------------------------------------------------------

class PredicateBlocker:
    """Generate candidate pairs via predicate-based blocking keys.

    For every predicate function the blocker builds an *inverted index*
    mapping each blocking key to the set of entity IDs that produced it.
    At query time the S1 entity's keys are looked up in the index to
    retrieve all matching S2/S3 entity IDs (union across predicates).

    Parameters
    ----------
    predicates : list[Callable] | None
        Predicate functions with signature ``(name, country) -> key | None``.
        When *None* the :func:`_build_default_predicates` set is used.
    """

    def __init__(
        self,
        predicates: list[Callable[..., str | None]] | None = None,
    ) -> None:
        self.predicates = predicates or _build_default_predicates()
        # One inverted index per predicate: key -> set[entity_id]
        self._indices: list[dict[str, set[str]]] = []

    # ----- index building ---------------------------------------------------

    def build_index(
        self,
        df: pd.DataFrame,
        name_col: str = "norm_name",
        country_col: str = "country",
        id_col: str = "entity_id",
    ) -> None:
        """Build inverted indices for the *reference* side (S2 + S3).

        Parameters
        ----------
        df : DataFrame
            Combined S2 + S3 data with at least *id_col*, *name_col* and
            *country_col*.
        """
        self._indices = [defaultdict(set) for _ in self.predicates]
        names = df[name_col].values
        countries = df[country_col].values
        ids = df[id_col].values

        for i in tqdm(range(len(df)), desc="PredicateBlocker · indexing", leave=False):
            name = str(names[i]) if pd.notna(names[i]) else ""
            country = str(countries[i]) if pd.notna(countries[i]) else ""
            eid = str(ids[i])
            for pidx, pred in enumerate(self.predicates):
                key = pred(name, country)
                if key is not None:
                    self._indices[pidx][key].add(eid)

        for pidx, pred in enumerate(self.predicates):
            logger.info(
                "PredicateBlocker [%s]: %d unique keys",
                getattr(pred, "__qualname__", str(pidx)),
                len(self._indices[pidx]),
            )

    # ----- querying ---------------------------------------------------------

    def query(
        self,
        df: pd.DataFrame,
        name_col: str = "norm_name",
        country_col: str = "country",
        id_col: str = "entity_id",
    ) -> dict[str, set[str]]:
        """Look up candidates for every entity in *df* (the S1 side).

        Returns
        -------
        dict[str, set[str]]
            Mapping ``{s1_entity_id: {candidate_entity_ids}}``.
        """
        if not self._indices:
            raise RuntimeError("Call build_index() before query().")

        candidates: dict[str, set[str]] = defaultdict(set)
        names = df[name_col].values
        countries = df[country_col].values
        ids = df[id_col].values

        for i in tqdm(range(len(df)), desc="PredicateBlocker · querying", leave=False):
            name = str(names[i]) if pd.notna(names[i]) else ""
            country = str(countries[i]) if pd.notna(countries[i]) else ""
            eid = str(ids[i])
            for pidx, pred in enumerate(self.predicates):
                key = pred(name, country)
                if key is not None and key in self._indices[pidx]:
                    candidates[eid].update(self._indices[pidx][key])

        return dict(candidates)


# ---------------------------------------------------------------------------
# TokenBlocker
# ---------------------------------------------------------------------------

class TokenBlocker:
    """Token-overlap blocking weighted by inverse document frequency.

    Each entity's normalised name is split into tokens. An inverted index
    maps every token to the entities containing it.  At query time, S2/S3
    entities that share at least *min_shared_tokens* tokens with the S1
    entity are returned as candidates.

    Tokens appearing in more than *max_token_freq* entities are pruned —
    they are too common to be discriminative and would cause a combinatorial
    explosion.

    Parameters
    ----------
    min_shared_tokens : int
        Minimum number of shared tokens to consider a pair a candidate.
    max_token_freq : int
        Tokens appearing in more than this many entities are skipped.
    """

    def __init__(
        self,
        min_shared_tokens: int = 1,
        max_token_freq: int = 5000,
    ) -> None:
        self.min_shared_tokens = min_shared_tokens
        self.max_token_freq = max_token_freq
        self._index: dict[str, set[str]] = defaultdict(set)
        self._idf: dict[str, float] = {}
        self._n_docs: int = 0

    # ----- index building ---------------------------------------------------

    def build_index(
        self,
        df: pd.DataFrame,
        name_col: str = "norm_name",
        id_col: str = "entity_id",
    ) -> None:
        """Build the token inverted index from the reference side (S2 + S3).

        Parameters
        ----------
        df : DataFrame
            Combined S2 + S3 data.
        """
        import math

        raw_index: dict[str, set[str]] = defaultdict(set)
        names = df[name_col].values
        ids = df[id_col].values
        self._n_docs = len(df)

        for i in tqdm(range(len(df)), desc="TokenBlocker · indexing", leave=False):
            name = str(names[i]) if pd.notna(names[i]) else ""
            eid = str(ids[i])
            tokens = set(name.split())
            for tok in tokens:
                if len(tok) >= 2:  # skip single-char noise
                    raw_index[tok].add(eid)

        # Prune overly-frequent tokens and compute IDF for the rest.
        self._index = defaultdict(set)
        self._idf = {}
        pruned = 0
        for tok, eids in raw_index.items():
            if len(eids) > self.max_token_freq:
                pruned += 1
                continue
            self._index[tok] = eids
            self._idf[tok] = math.log((self._n_docs + 1) / (len(eids) + 1)) + 1.0

        logger.info(
            "TokenBlocker: %d tokens indexed, %d pruned (freq > %d)",
            len(self._index),
            pruned,
            self.max_token_freq,
        )

    # ----- querying ---------------------------------------------------------

    def query(
        self,
        df: pd.DataFrame,
        name_col: str = "norm_name",
        id_col: str = "entity_id",
    ) -> dict[str, set[str]]:
        """Find candidates for every S1 entity via token overlap.

        Returns
        -------
        dict[str, set[str]]
            ``{s1_entity_id: {candidate_entity_ids}}``.
        """
        if not self._index:
            raise RuntimeError("Call build_index() before query().")

        candidates: dict[str, set[str]] = defaultdict(set)
        names = df[name_col].values
        ids = df[id_col].values

        for i in tqdm(range(len(df)), desc="TokenBlocker · querying", leave=False):
            name = str(names[i]) if pd.notna(names[i]) else ""
            eid = str(ids[i])
            tokens = set(name.split())

            if self.min_shared_tokens <= 1:
                # Fast path: any shared token is enough.
                for tok in tokens:
                    if tok in self._index:
                        candidates[eid].update(self._index[tok])
            else:
                # Count shared tokens per candidate and threshold.
                overlap_count: dict[str, int] = defaultdict(int)
                for tok in tokens:
                    if tok in self._index:
                        for cid in self._index[tok]:
                            overlap_count[cid] += 1
                for cid, cnt in overlap_count.items():
                    if cnt >= self.min_shared_tokens:
                        candidates[eid].add(cid)

        return dict(candidates)


# ---------------------------------------------------------------------------
# Country filter helper
# ---------------------------------------------------------------------------

def _apply_country_filter(
    candidates: dict[str, set[str]],
    s1_countries: dict[str, str],
    ref_countries: dict[str, str],
) -> dict[str, set[str]]:
    """Remove candidate pairs where the countries do not match.

    Parameters
    ----------
    candidates : dict
        ``{s1_id: set_of_candidate_ids}``
    s1_countries : dict
        ``{entity_id: country}`` for S1 entities.
    ref_countries : dict
        ``{entity_id: country}`` for S2 + S3 entities.

    Returns
    -------
    dict[str, set[str]]
        Filtered candidate mapping.
    """
    filtered: dict[str, set[str]] = {}
    for s1_id, cand_ids in candidates.items():
        s1_country = s1_countries.get(s1_id, "")
        if not s1_country:
            # Unknown country — keep all candidates.
            filtered[s1_id] = cand_ids
            continue
        kept = {
            cid for cid in cand_ids
            if ref_countries.get(cid, "") == s1_country
            or not ref_countries.get(cid, "")
        }
        if kept:
            filtered[s1_id] = kept
    return filtered


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prune_candidates_by_name(
    candidates: dict[str, set[str]],
    s1_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    name_col: str = "norm_name",
    id_col: str = "entity_id",
    min_score: float = 0.55,
    max_per_s1: int = 200,
) -> dict[str, set[str]]:
    """Cheap second-stage filter using rapidfuzz token-set ratio.

    Keeps up to *max_per_s1* highest-scoring candidates per S1 entity whose
    name similarity is at least *min_score*. This is the set fed to the
    full feature / ML stage (and therefore what belongs in
    ``candidate_pairs.tsv``).
    """
    from rapidfuzz import fuzz as rfuzz

    s1_names = dict(zip(s1_df[id_col].astype(str), s1_df[name_col].fillna("").astype(str)))
    ref_names = dict(zip(ref_df[id_col].astype(str), ref_df[name_col].fillna("").astype(str)))

    pruned: dict[str, set[str]] = {}
    for s1_id, cands in tqdm(candidates.items(), desc="Pruning candidates", leave=False):
        s1_name = s1_names.get(s1_id, "")
        if not s1_name or not cands:
            continue

        scored: list[tuple[float, str]] = []
        for cid in cands:
            cname = ref_names.get(cid, "")
            if not cname:
                continue
            score = rfuzz.token_set_ratio(s1_name, cname) / 100.0
            if score >= min_score:
                scored.append((score, cid))

        if not scored:
            # Keep a few best even below threshold so recall does not collapse.
            fallback: list[tuple[float, str]] = []
            for cid in cands:
                cname = ref_names.get(cid, "")
                if not cname:
                    continue
                fallback.append((rfuzz.token_set_ratio(s1_name, cname) / 100.0, cid))
            fallback.sort(reverse=True)
            pruned[s1_id] = {cid for _, cid in fallback[: min(20, max_per_s1)]}
            continue

        scored.sort(reverse=True)
        pruned[s1_id] = {cid for _, cid in scored[:max_per_s1]}

    return pruned


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    *,
    name_col: str = "norm_name",
    country_col: str = "country",
    id_col: str = "entity_id",
    min_shared_tokens: int = 1,
    max_token_freq: int = 5000,
    country_filter: bool = True,
    predicate_predicates: list[Callable[..., str | None]] | None = None,
    batch_size: int = 500_000,
    prune: bool = True,
    prune_min_score: float = 0.55,
    prune_max_per_s1: int = 100,
) -> dict[str, set[str]]:
    """Combine all blocking strategies and return unified candidates.

    The function merges Source 2 and Source 3 into a single reference set,
    builds blocking indices once, and then queries them with Source 1.
    An optional name-similarity prune produces the final candidate set
    that is scored by the matcher (and written to ``candidate_pairs.tsv``).
    """
    logger.info(
        "generate_candidates: S1=%d, S2=%d, S3=%d rows",
        len(s1_df), len(s2_df), len(s3_df),
    )

    ref_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info("Reference set (S2+S3): %d rows", len(ref_df))

    total_possible_pairs = len(s1_df) * len(ref_df)

    pred_blocker = PredicateBlocker(predicates=predicate_predicates)
    pred_blocker.build_index(ref_df, name_col=name_col, country_col=country_col, id_col=id_col)

    tok_blocker = TokenBlocker(
        min_shared_tokens=min_shared_tokens,
        max_token_freq=max_token_freq,
    )
    tok_blocker.build_index(ref_df, name_col=name_col, id_col=id_col)

    all_candidates: dict[str, set[str]] = {}
    n_batches = (len(s1_df) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(s1_df))
        s1_batch = s1_df.iloc[start:end]
        logger.info("Processing S1 batch %d/%d (%d–%d)", batch_idx + 1, n_batches, start, end)

        pred_cands = pred_blocker.query(s1_batch, name_col=name_col, country_col=country_col, id_col=id_col)
        tok_cands = tok_blocker.query(s1_batch, name_col=name_col, id_col=id_col)

        batch_ids = s1_batch[id_col].astype(str).values
        for eid in batch_ids:
            merged = set()
            if eid in pred_cands:
                merged.update(pred_cands[eid])
            if eid in tok_cands:
                merged.update(tok_cands[eid])
            if merged:
                all_candidates[eid] = merged

    if country_filter:
        s1_countries = dict(
            zip(
                s1_df[id_col].astype(str),
                s1_df[country_col].fillna("").astype(str),
            )
        )
        ref_countries = dict(
            zip(
                ref_df[id_col].astype(str),
                ref_df[country_col].fillna("").astype(str),
            )
        )
        all_candidates = _apply_country_filter(all_candidates, s1_countries, ref_countries)

    raw_total = sum(len(v) for v in all_candidates.values())
    print(f"  Raw blocked pairs (pre-prune): {raw_total:,}")

    if prune:
        all_candidates = prune_candidates_by_name(
            all_candidates,
            s1_df,
            ref_df,
            name_col=name_col,
            id_col=id_col,
            min_score=prune_min_score,
            max_per_s1=prune_max_per_s1,
        )

    total_cands = sum(len(v) for v in all_candidates.values())
    n_s1_with_cands = len(all_candidates)
    avg_cands = total_cands / max(n_s1_with_cands, 1)
    reduction = 1.0 - (total_cands / total_possible_pairs) if total_possible_pairs else 0.0

    stats_msg = (
        f"\n{'=' * 60}\n"
        f"Blocking statistics\n"
        f"{'=' * 60}\n"
        f"  S1 entities with ≥1 candidate : {n_s1_with_cands:>12,}\n"
        f"  S1 entities without candidates : {len(s1_df) - n_s1_with_cands:>12,}\n"
        f"  Total candidate pairs          : {total_cands:>12,}\n"
        f"  Avg candidates per S1 entity   : {avg_cands:>12,.1f}\n"
        f"  Total possible pairs           : {total_possible_pairs:>12,}\n"
        f"  Reduction ratio                : {reduction:>12.6f}\n"
        f"{'=' * 60}"
    )
    print(stats_msg)
    logger.info(stats_msg)

    return all_candidates
