# Public Repository Structure

This public repository is a release package for the COMBELT-2026 manuscript. It is not the original raw-crawl research workspace.

```text
data/
  processed/          Curated clause-level dataset
  splits/             Review-level split map
  publication/        Summary files used for manuscript tables
  data_dictionary/    Field descriptions and data-release notes

outputs/
  evaluation/         Archived evaluation reports and predictions
  publication_tables/ Reserved for exported paper tables
  logs/               Reserved for release/run logs

code/
  ssot_build/         Dataset and publication-analytics scripts
  aspect_screening/   Aspect evaluation scripts
  sentiment_visobert/ Sentiment diagnostic scripts
  bertopic/           Exploratory BERTopic scripts
  shap_pilot/         Post-hoc SHAP pilot scripts
  utils/              Shared path helpers
  verify_release.py   Public-release hygiene check

docs/
  methodology/        Method notes
  reproducibility/    Release verification and run guidance
  figures_tables/     Reserved for exported figures and tables
```

## Public Entry Points

| Purpose | Path |
|---|---|
| Main curated clause dataset | `data/processed/clauses_final_43c_44_20260528_034911.csv` |
| Review-level split map | `data/splits/downstream_split_reviewid_20260528_022322.csv` |
| Publication summary files | `data/publication/phase55_final_001/` |
| T2 aspect holdout outputs | `outputs/evaluation/aspect_t2_gold_holdout/` |
| T3 sentiment primary outputs | `outputs/evaluation/sentiment_t3_primary_no_adrd25/` |
| T3 sensitivity outputs | `outputs/evaluation/sentiment_t3_sensitivity/` |
| Post-hoc SHAP pilot outputs | `outputs/evaluation/shap_pilot_phase55_final_001/` |
| Release hygiene check | `code/verify_release.py` |

## Release Boundary

The raw platform crawl, private metadata, temporary work files, model binaries, and manuscript drafts are intentionally excluded. The public package contains processed research data, archived outputs, and method code sufficient to audit the reported manuscript numbers.
