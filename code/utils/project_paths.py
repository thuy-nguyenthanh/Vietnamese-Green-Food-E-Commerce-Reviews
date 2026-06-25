"""Canonical paths for the public reproducibility package."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
CODE_DIR = REPO_ROOT / "code"
DOCS_DIR = REPO_ROOT / "docs"
OUTPUTS_DIR = REPO_ROOT / "outputs"

PROCESSED_DATA = DATA_DIR / "processed"
SPLITS_DATA = DATA_DIR / "splits"
PUBLICATION_DATA = DATA_DIR / "publication"
DATA_DICTIONARY = DATA_DIR / "data_dictionary"

EVALUATION_OUTPUTS = OUTPUTS_DIR / "evaluation"
PUBLICATION_TABLES = OUTPUTS_DIR / "publication_tables"
LOGS = OUTPUTS_DIR / "logs"

MAIN_CLAUSES = PROCESSED_DATA / "clauses_final_43c_44_20260528_034911.csv"
REVIEW_SPLIT_MAP = SPLITS_DATA / "downstream_split_reviewid_20260528_022322.csv"

ASPECT_T2_HOLDOUT = EVALUATION_OUTPUTS / "aspect_t2_gold_holdout"
SENTIMENT_T3_PRIMARY = EVALUATION_OUTPUTS / "sentiment_t3_primary_no_adrd25"
SENTIMENT_T3_SENSITIVITY = EVALUATION_OUTPUTS / "sentiment_t3_sensitivity"
SHAP_PILOT = EVALUATION_OUTPUTS / "shap_pilot_phase55_final_001"
