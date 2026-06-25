# Vietnamese Green-Food E-Commerce Reviews

Code and data for an audit-governed text analytics study of Vietnamese green-food e-commerce reviews.

The repository supports the paper **"Audit-Governed Text Analytics for SME Monitoring of Vietnamese Green-Food E-Commerce Reviews"** submitted to COMBELT-2026.

## What This Repository Contains

This is a reproducibility package, not a raw crawl dump. It contains processed data, archived evaluation outputs, scripts, and documentation needed to audit the main results reported in the paper. The raw crawl and private platform metadata are not redistributed.

Main contents:

- `data/processed/`: curated clause-level dataset used as the single source of truth.
- `data/splits/`: review-level split file used to reduce clause leakage.
- `data/publication/`: summary files used for manuscript tables and reported numbers.
- `outputs/evaluation/`: archived aspect, sentiment, and SHAP evaluation outputs.
- `code/`: scripts and release checks for dataset preparation, evaluation, and publication analytics.
- `docs/`: method notes and reproducibility guidance.
- `docs/reproducibility/LLM_ANNOTATION_PROMPT.md`: Step F LLM annotation prompt for silver-label generation.

## Study Boundary

The data come from Vietnamese Shopee reviews of seller-positioned green, organic, or healthy food products. The released corpus is a curated hard-case sample of **1,742 ABSA-eligible clauses**.

The data should not be read as:

- a market-wide estimate of Vietnamese e-commerce;
- a verified registry of certified-organic products;
- a stable topic taxonomy;
- causal evidence from SHAP values;
- evidence for fully automated deployment.

The aim is narrower: to provide auditable monitoring evidence for SME review screening.

## Label Provenance

The workflow separates label sources:

- **Silver labels**: LLM-assisted labels used for descriptive analysis.
- **Human-gold labels**: audited labels used for headline evaluation.
- **Evaluation splits**: grouped by `review_id` to reduce clause leakage.

Please do not treat silver labels as human-gold labels.
The LLM prompt used for the silver-label pass is archived in
`docs/reproducibility/LLM_ANNOTATION_PROMPT.md`.

## Repository Layout

```text
data/
  processed/          Curated clause-level dataset
  splits/             Review-level train/validation/test splits
  publication/        Summary files for manuscript numbers
  data_dictionary/    Field descriptions

code/
  ssot_build/         Dataset, funnel, descriptive, and bootstrap scripts
  aspect_screening/   Aspect evaluation scripts
  sentiment_visobert/ ViSoBERT diagnostic scripts
  bertopic/           BERTopic preparation / exploratory scripts
  shap_pilot/         Post-hoc exploratory SHAP pilot
  utils/              Shared path helpers

outputs/
  evaluation/         Reproduced evaluation reports
  publication_tables/ Reserved for table exports
  logs/               Reserved for run logs

docs/
  methodology/        Method notes
  reproducibility/    Run order and checks
  figures_tables/     Reserved for exported figures/tables
```

## Main Results Covered

The package covers:

- curated hard-case corpus: `n = 1,742` clauses;
- T2 aspect holdout evaluation: Macro-F1 about `0.458`;
- T3 out-of-domain ViSoBERT sentiment diagnosis: Macro-F1 about `0.5074`;
- exploratory BERTopic outputs;
- post-hoc SHAP pilot for review-level rating signals.

Exact numbers should be taken from the files under `outputs/evaluation/` and `data/publication/`.

## Release Check

Before publishing or tagging a release, run:

```bash
python code/verify_release.py
```

The check verifies expected row counts, scans released text fields for direct identifiers such as phone-like strings, checks JSON validity, and flags absolute local workspace paths or secret-like tokens.

## Quick Start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

See `docs/reproducibility/RUN_ORDER.md` for the recommended verification order. Some scripts are methodological snapshots from the original research workspace; the released package prioritizes auditable processed data and archived outputs rather than redistributing the full raw-crawl pipeline.

## Ethics and Data Availability

The released data are processed research data derived from publicly visible product reviews. Usernames, seller identifiers, direct identifiers, and phone-like strings are removed or masked in the public package. Raw platform crawl files are not redistributed here.

All released code and data are provided at:

https://github.com/thuy-nguyenthanh/Vietnamese-Green-Food-E-Commerce-Reviews

## Citation

If you use this repository, please cite the accompanying paper:

```text
Nguyen-Thanh, T., Le, T. N. N., Phan, T. T., Truong, T. L.,
Pham, T. P., & Nguyen, H. Q. (2026).
Audit-Governed Text Analytics for SME Monitoring of Vietnamese
Green-Food E-Commerce Reviews. COMBELT-2026.
```

Please update the citation after proceedings details are available.

## License

See `LICENSE`. Code and data may have different reuse conditions.
