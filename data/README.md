# Data provenance and corrections

`indicator_dictionary.csv` records the 34 operational definitions and codes of manuscript Table 1, alongside the exact model-column names.

`raw/` contains the six annual workbooks used in the analysis. The 2023 source is the corrected `23年小德硬地技术统计-修正.xlsx`. There were no eligible 2022 Djokovic Grand Slam hard-court matches under the study's inclusion criteria. Annotator fields and author metadata in all six workbooks have been de-identified ("Study team"); the source map and checksums retain traceability.

`processed/train_grouped.csv` and `processed/test_grouped.csv` are the authoritative English-labelled model inputs. Each row is one set. The first column, `Set Result`, is the binary outcome. Columns 2–35 are the 34 candidate indicators; the remaining columns document match identity, year, event, round, within-match set number, source row, validation group and the second-serve-speed correction. All sets from the same `match_id` stay within one group during validation. `reference/match_mapping.csv` identifies the source workbook and column for every set. The 2021 workbook has a one-row-shifted feature layout, which `analysis/build_processed.py` handles explicitly.

The corrected 2024 source workbook records **155 km/h** at `Sheet1!Y26` for second-serve speed, replacing the erroneous 455. `Net Points Won` is the name of the points-won-at-the-net indicator (X30) in all workbooks, tables and arrays. The 2023 forecourt/midcourt/backcourt depth proportions were renormalized to sum to 1.00 in each set. Three other corrected depth rows are documented in `reference/data_verification.json`: 2020 Australian Open final set 3; 2025 Australian Open quarterfinal set 1; and 2025 US Open first-round set 2. Other years retain their original two-decimal rounding, which can yield totals of 0.99 or 1.01.

The train and test tables contain 199 and 40 sets: 163 winning and 36 losing sets from 59 development matches, and 30 winning and 10 losing sets from 12 matches in 2025. `analysis/verify_data.py` checks the workbooks and tables against `sha256.csv`, row counts, group counts, depth totals and the 155 km/h entry. `analysis/build_processed.py` rebuilds every numerical input from the workbooks and verifies it against the deposited tables. Four serve-rate values were rounded to two decimals in the model table and three depth triples were renormalized as described above; `analysis/audit_source_mapping.py` lists these 11 source-to-table cell differences.

## Reference records for the supplementary tables

`reference/` holds the deposited predictions, member probabilities and SHAP explanations of the primary workflow (Tables S2–S5, Figures 3–5), the classification comparators and F1-selection workflow (`outer_predictions.csv`, `test_predictions.csv`), the hyperparameter-tuning sensitivity (`tuning_*`, `tuned_member_shap/`) and the twelve-indicator exclusion (`twelve_feature_*`, `revision2026/twelve_member_shap/`, `revision2026/twelve_agreement/`).

`reference/revision2026/` holds the records behind Tables S3 (2025 interval metrics), S5 (Figure 4b ablation), S6 (four-group agreement), S8 (opponent adjustment), S9 (court-depth omission), S10 (share and rate representations, training-mean mask) and S11 (two-observer recoding).

`revision2026/reliability/observer_records.csv` transcribes the S11 records: original and both observer values for 34 indicators plus outcome across six sets from two development matches (recoding date 18 November 2026; recorded selection seed 2026). The pooled count kappa combines 120 indicator–set pairs; ICC is calculated within indicators. Eight tracking-derived measures are kept in the record but excluded from the agreement summary. No observer identities are deposited.

`revision2026/reliability/external_match_pairs.csv` is the author-aligned 36-match comparison with the public Kaggle **ATP matches** dataset, version 3 (https://www.kaggle.com/datasets/sijovm/atpdata/versions/3), created by aligning the author's filtered intermediate file `小德公开数据集（英文）.csv` to `atp_matches_till_2022.csv`. The four compared counts are Aces, Double Faults, Break Points and Service Breaks. It is a third-party match-data check, not a claim of direct ATP website extraction.

Opponent ATP rank (Table S8) was obtained from the pinned public mirror of Jeff Sackmann's match data at https://huggingface.co/datasets/jmatarese/atp-data/tree/c704c1bcbe08949246d956b54b099715d5997dcb . Upstream README, data dictionary and download manifest are retained under `reference/revision2026/opponent/sources/`.

## Third-party licence and source lineage

The Kaggle ATP matches v3 metadata (checked 19 November 2026) labels its licence "Other (specified in description)" and credits Jeff Sackmann and Tennis Abstract. The pinned opponent-data mirror preserves Sackmann's CC BY-NC-SA 4.0 notice. The comparator records and opponent metadata therefore share a source lineage; they are external recording checks, not independent validation against two unrelated providers. Source-derived comparator columns and `opponent_metadata_71_matches.csv` / `opponent_metadata_239_sets.csv` retain attribution to Jeff Sackmann / Tennis Abstract and CC BY-NC-SA 4.0.

The six complete upstream annual CSVs are not redistributed. Their immutable URLs and SHA-256 values are in the download manifest; `python analysis/fetch_opponent_sources.py` optionally retrieves them. Reproduction and opponent-model refitting use the deposited, attributed 239-set metadata.
