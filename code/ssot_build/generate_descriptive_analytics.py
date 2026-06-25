"""
Descriptive analytics on extended pool (silver pipeline labels, n=1742).

Outputs under Step05_SSOT/Datas/publication/<run_tag>/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
INPUT_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "downstream_inputs"
OUT_BASE = WORKSPACE / "Step05_SSOT" / "Datas" / "publication"
REFRESH_PATH = WORKSPACE / "Step05_SSOT" / "Datas" / "phase55" / "phase55_final_001" / "refresh_summary_before_after_neg.json"
COHORT_B_NEG_TARGET = 350
COHORT_B_TRAIN_NEG_TARGET = 280


def _load_extended_pool(run_tag: str) -> pd.DataFrame:
    base = INPUT_BASE / run_tag
    parts = []
    for split in ("train", "val", "test"):
        path = base / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["split"] = split
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)
    return pool


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def _distribution(series: pd.Series, total: int) -> dict:
    counts = series.astype(str).value_counts()
    return {k: {"n": int(v), "pct": _pct(int(v), total)} for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate descriptive analytics (silver labels).")
    parser.add_argument("--run-tag", default="phase55_final_001")
    args = parser.parse_args()

    pool = _load_extended_pool(args.run_tag)
    n = len(pool)
    if n != 1742:
        print(f"Warning: expected n=1742, got n={n}")

    aspect_col = "aspect_hint"
    sentiment_col = "sentiment_polarity_hint"

    crosstab = pd.crosstab(pool[aspect_col], pool[sentiment_col], margins=True)
    crosstab_pct_row = pd.crosstab(pool[aspect_col], pool[sentiment_col], normalize="index").round(4)

    neg = pool.loc[pool[sentiment_col].astype(str) == "NEG"].copy()
    neg_by_aspect = neg[aspect_col].astype(str).value_counts()
    neg_aspect_rows = [
        {"aspect": a, "n": int(c), "pct_of_neg": _pct(int(c), len(neg))} for a, c in neg_by_aspect.items()
    ]

    if "complaint_severity" in neg.columns:
        sev = neg.groupby(aspect_col)["complaint_severity"].value_counts().unstack(fill_value=0)
        sev.to_csv(
            OUT_BASE / args.run_tag / "neg_complaint_severity_by_aspect.csv",
            encoding="utf-8-sig",
        )

    refresh = {}
    if REFRESH_PATH.exists():
        refresh = json.loads(REFRESH_PATH.read_text(encoding="utf-8"))

    out_dir = OUT_BASE / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    crosstab.to_csv(out_dir / "aspect_sentiment_crosstab.csv", encoding="utf-8-sig")
    crosstab_pct_row.to_csv(out_dir / "aspect_sentiment_crosstab_row_pct.csv", encoding="utf-8-sig")
    pd.DataFrame(neg_aspect_rows).to_csv(out_dir / "neg_complaint_by_aspect.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "label_type": "silver_pipeline_labels",
        "label_columns": {"aspect": aspect_col, "sentiment": sentiment_col},
        "n_clauses": n,
        "D1_aspect_distribution": _distribution(pool[aspect_col], n),
        "D2_sentiment_distribution": _distribution(pool[sentiment_col], n),
        "D3_aspect_sentiment_crosstab_files": [
            "aspect_sentiment_crosstab.csv",
            "aspect_sentiment_crosstab_row_pct.csv",
        ],
        "D4_neg_complaint_by_aspect": neg_aspect_rows,
        "D5_annotation_source": _distribution(pool["annotation_source"], n),
        "D6_partial_cohort_b": {
            "neg_extended": int((pool[sentiment_col].astype(str) == "NEG").sum()),
            "neg_extended_target": COHORT_B_NEG_TARGET,
            "train_neg_after_tier2": int(refresh.get("train_neg_after", 212)),
            "train_neg_target": COHORT_B_TRAIN_NEG_TARGET,
        },
        "manuscript_note": (
            "Descriptive tables use silver pipeline labels after LLM-assisted preprocessing "
            "and partial human QC — not human gold audit lanes (T2/T3)."
        ),
    }

    summary_path = out_dir / "descriptive_analytics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
