"""
Publication funnel + cohort summary for Phase G (tech-econ framing).

Outputs under Step05_SSOT/Datas/publication/<run_tag>/:
  - funnel_table.json
  - cohort_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
PROFILE_PATH = WORKSPACE / "Step01_31" / "Datas" / "raw_reviews" / "_profile_31_32.json"
CORE_PATH = WORKSPACE / "Step04_41_43c" / "Datas" / "43c_llm_assisted" / "batches" / "validated" / "clauses_core_all_v3.csv"
SSOT_DIR = WORKSPACE / "Step05_SSOT" / "Datas" / "clauses_ssot"
PHASE55_BASE = WORKSPACE / "Step05_SSOT" / "Datas" / "phase55"
GOLD_T2 = (
    WORKSPACE
    / "Step04_41_43c"
    / "Datas"
    / "43c_llm_assisted"
    / "stepG_human_review"
    / "iaa_batches"
    / "stepG_gold_labels_v1.csv"
)
OUT_BASE = WORKSPACE / "Step05_SSOT" / "Datas" / "publication"
COHORT_B_NEG_TARGET = 350
COHORT_B_TRAIN_NEG_TARGET = 280


def _read_latest_ssot() -> Path:
    latest = SSOT_DIR / "LATEST.txt"
    if not latest.exists():
        raise FileNotFoundError(f"Missing {latest}")
    name = latest.read_text(encoding="utf-8").strip().splitlines()[0]
    path = SSOT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication funnel + cohort summary.")
    parser.add_argument("--run-tag", default="phase55_final_001")
    parser.add_argument("--ssot-csv", default=None)
    args = parser.parse_args()

    ssot_path = Path(args.ssot_csv) if args.ssot_csv else _read_latest_ssot()
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    core = pd.read_csv(CORE_PATH, low_memory=False)
    ssot = pd.read_csv(ssot_path, low_memory=False)
    ext = ssot.loc[_bool_col(ssot["has_extended_flags"])].copy()

    nhr = _bool_col(core["need_human_review"])
    absa = _bool_col(core["keep_for_absa"])

    # v2 documented need_llm_assisted ≈1166; verify from core if column exists
    llm_reviews_doc = 1166
    if "need_llm_assisted" in core.columns:
        llm_mask = _bool_col(core["need_llm_assisted"])
        llm_reviews_computed = int(core.loc[llm_mask, "review_id"].nunique())
    else:
        llm_reviews_computed = None

    core_reviews = int(core["review_id"].nunique())
    ext_reviews = int(ext["review_id"].nunique())

    tier3_iaa = json.loads(
        (PHASE55_BASE / "phase55_tier23_001" / "tier3" / "analysis" / "iaa_report.json").read_text(
            encoding="utf-8"
        )
    )
    refresh = json.loads(
        (PHASE55_BASE / args.run_tag / "refresh_summary_before_after_neg.json").read_text(encoding="utf-8")
    )

    t2_gold = pd.read_csv(GOLD_T2)
    t2_n = int(t2_gold["clause_id_final"].notna().sum())

    funnel_stages = [
        {
            "stage_id": "S1_crawl",
            "label": "Raw reviews crawled (Shopee)",
            "n": int(profile["n_reviews"]),
            "unit": "review",
            "source": str(PROFILE_PATH),
        },
        {
            "stage_id": "S2_human_qc",
            "label": "After human QC 3.2c (documented v2)",
            "n": 126329,
            "unit": "review",
            "source": "Giai_phap_Hybrid_Pipeline_BERTopic_ViSoBERT_v2.md §3.2c",
            "note": "Static Methods count; verify against Step02 if updated",
        },
        {
            "stage_id": "S3_clause_43b",
            "label": "Clauses after NLP segmentation 4.3b",
            "n": 176613,
            "unit": "clause",
            "source": "Giai_phap v2/v3 §2.1",
        },
        {
            "stage_id": "S4_llm_hardcase",
            "label": "LLM-assisted hard-case reviews (4.3c path)",
            "n": llm_reviews_computed if llm_reviews_computed is not None else llm_reviews_doc,
            "unit": "review",
            "source": "core need_llm_assisted or v2 documented 1166",
            "documented_n": llm_reviews_doc,
        },
        {
            "stage_id": "S5_core_clauses",
            "label": "Core clauses after 4.3c A–E",
            "n": len(core),
            "unit": "clause",
            "n_reviews": core_reviews,
            "source": str(CORE_PATH),
        },
        {
            "stage_id": "S6_extended_absa",
            "label": "ABSA-eligible extended pool (Step F + Phase 5.5 partial)",
            "n": len(ext),
            "unit": "clause",
            "n_reviews": ext_reviews,
            "source": str(ssot_path),
        },
        {
            "stage_id": "S7_audit_t2",
            "label": "Human aspect gold audit (T2)",
            "n": t2_n,
            "unit": "clause",
            "kappa": "~0.95",
            "source": str(GOLD_T2),
        },
        {
            "stage_id": "S8_audit_t3",
            "label": "Human sentiment mini-IAA (T3)",
            "n": int(tier3_iaa.get("n_rows", 120)),
            "unit": "clause",
            "kappa": tier3_iaa.get("kappa_overall"),
            "source": "phase55_tier23_001/tier3/analysis/iaa_report.json",
        },
    ]

    sentiment_counts = ext["sentiment_polarity_hint"].astype(str).value_counts().to_dict()
    aspect_counts = ext["aspect_hint"].astype(str).value_counts().to_dict()
    annotation_counts = ext["annotation_source"].astype(str).value_counts().to_dict()

    neg_extended = int((ext["sentiment_polarity_hint"].astype(str) == "NEG").sum())

    cohort_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "ssot_csv": str(ssot_path),
        "extended_pool_n": len(ext),
        "extended_reviews_n": ext_reviews,
        "core_counts": {
            "clauses": len(core),
            "reviews": core_reviews,
            "keep_for_absa": int(absa.sum()),
            "need_human_review": int(nhr.sum()),
            "absa_and_nhr": int((absa & nhr).sum()),
        },
        "sentiment_distribution": {
            k: {"n": int(v), "pct": _pct(int(v), len(ext))} for k, v in sentiment_counts.items()
        },
        "aspect_distribution": {
            k: {"n": int(v), "pct": _pct(int(v), len(ext))} for k, v in aspect_counts.items()
        },
        "annotation_source": {
            k: {"n": int(v), "pct": _pct(int(v), len(ext))} for k, v in annotation_counts.items()
        },
        "partial_cohort_b": {
            "neg_extended": neg_extended,
            "neg_extended_target": COHORT_B_NEG_TARGET,
            "neg_extended_pct_of_target": _pct(neg_extended, COHORT_B_NEG_TARGET),
            "train_neg_after_tier2": int(refresh.get("train_neg_after", 212)),
            "train_neg_target": COHORT_B_TRAIN_NEG_TARGET,
        },
        "phase55_audit_lanes": {
            "tier1_neg_qc": 18,
            "tier2_spotcheck_pass": 11,
            "tier3_mini_iaa_n": 120,
            "t2_aspect_gold_n": t2_n,
        },
    }

    out_dir = OUT_BASE / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    funnel_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "framing": "big_data_funnel_to_curated_absa_corpus",
        "stages": funnel_stages,
        "claim_boundary": (
            "Supervised and descriptive analytics apply to ABSA-eligible hard-case clauses, "
            "not the full 127k review crawl."
        ),
    }

    funnel_path = out_dir / "funnel_table.json"
    cohort_path = out_dir / "cohort_summary.json"
    funnel_path.write_text(json.dumps(funnel_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cohort_path.write_text(json.dumps(cohort_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"funnel": str(funnel_path), "cohort": str(cohort_path)}, indent=2))


if __name__ == "__main__":
    main()
