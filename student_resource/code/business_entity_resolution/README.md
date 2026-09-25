# Business Entity Resolution

End-to-end pipeline for the ML Challenge 2026 Business Entity Resolution task.

## Approach

1. **Preprocessing** — transliteration (Indic → Latin / unidecode), abbreviation expansion (`pvt`→`private`, etc.), bracket/punctuation stripping, country-aware address standardization, and structured address parsing for US, India, and France/generic.
2. **Blocking** — union of predicate keys (name prefixes, sorted tokens, Soundex, country+prefix) and token-overlap blocking, with an optional same-country filter.
3. **Features** — name similarities (Levenshtein, affine-gap/Indel, Jaro-Winkler, token sort/set/partial, Jaccard, overlap, IDF cosine), phonetic (Soundex/Metaphone), address score = normalised sum of raw Levenshtein + token overlap + city/state/pin matches, plus a name-heavy weighted blend.
4. **Matching** — XGBoost classifier with F₀.₅-tuned threshold (default), or a similarity-threshold fallback (`--matcher similarity`).

## Setup

```bash
cd student_resource
pip install -r code/business_entity_resolution/requirements.txt
```

## Run

From `student_resource/`:

```bash
# Smoke test on a small subset (recommended first)
python code/business_entity_resolution/src/pipeline.py \
  --mode sample --sample-size 2000 --matcher xgboost

# Full train + test (writes output/matching_results.tsv + candidate_pairs.tsv)
python code/business_entity_resolution/src/pipeline.py --mode full

# Train only / test only
python code/business_entity_resolution/src/pipeline.py --mode train
python code/business_entity_resolution/src/pipeline.py --mode test --model-path output/model.pkl
```

Validate submission format:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Layout

```
src/
  preprocessing.py   # transliteration, standardization, address parse
  blocking.py        # predicate + token blocking
  features.py        # pairwise similarity features
  matching.py        # XGBoost / similarity matcher + F0.5 threshold tuning
  pipeline.py        # CLI orchestration
```

## Notes

- All I/O is tab-separated (`.tsv`).
- `country` is treated as an open set — France appears only in test and is handled via generic/France address parsing, not a closed one-hot over `{US, India}`.
- F₀.₅ is precision-heavy; the matcher threshold is tuned accordingly and name similarity is weighted above address.
- Blocking uses predicate keys + token overlap, then a rapidfuzz name prune (top 100 / S1, min token-set score 0.55) so the ML stage stays tractable at multi-million-row scale.
- On macOS, XGBoost needs OpenMP:

```bash
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

If XGBoost still fails to load, the pipeline falls back to `--matcher similarity`.
