## Plan: ML Entity Resolution Pipeline

I’d use a precision-focused hybrid ER pipeline: normalize noisy business names/addresses, generate a high-recall candidate set with blocking, then score candidate pairs with an ML matcher tuned for `F_0.5`. Because the leaderboard only scores matches from `Source 1` to `Source 2/3`, the plan should treat this as a bipartite pair-classification problem with careful thresholding, country-aware features, and strict submission validation via `utils/validate_submission.py` and the output rules in `README.md`.

### Steps
1. **Profile the data and label patterns** in `dataset/train/` and summarize them in `Documentation_template.md`, especially name/address noise, singleton rate, and country-specific differences.
2. Preprocess the data by normalizing it and transliterating.
2. **Design a normalization layer** for business names, addresses, and country labels, keeping `country` open-set so `France` and any future labels are preserved.
3. **Build high-recall blocking** over `Source 1` vs `Source 2/3` using lightweight keys plus text-similarity retrieval to create `candidate_pairs.tsv` Use Predicate Blocks with useful predicate functions such as "Same first 5 characters".
4. **Train a pairwise matching model** on candidate pairs with features for name, address, and metadata similarity; tune the decision threshold for `F_0.5`.
5. **Generate final outputs** in `output/matching_results.tsv` and `output/candidate_pairs.tsv`, ensuring every `Source 1` test entity appears exactly once.
6. **Validate and iterate** using `utils/validate_submission.py` and a held-out training split before finalizing the write-up and submission package.

### Further Considerations
1. Use a **hybrid approach**: rules-based blocking + ML classifier is usually safer than end-to-end matching for precision-heavy ER.
2. Prioritize **false-positive control**: missed matches hurt less than wrong merges under `F_0.5`.
3. Keep the methodology document aligned with the final pipeline so the package is reproducible under `code/business_entity_resolution/`.

preprocessing - 
    transilteration
    standardization-(pvt to private, brackets remove)
    normalization for names-(same case, remove punctuation)
    address can be parsed directly using libraries

blocking-
    predicate blocking-(first 5 characters, first 3 words, etc)
    token blocking-common name tokens

features:
    name tokens-(levenshtein and affine gap distance)
    phonetic(soundex) for each token
    different address parsing for India and US
    address score-(sum of raw address similarity, token overlap, and parsed components)
    More weight for name similarity than address similarity

Train using XGBoost or just use similarity score directly with a threshold tuned for F_0.5.
