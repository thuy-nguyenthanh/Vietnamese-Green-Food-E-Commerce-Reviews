"""
T3 sentiment error analysis: extract misclassified clauses for qualitative review.

Output: >=30 misclassified cases with gold/pred labels and error type tags.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
EVAL_BASE = WORKSPACE / "Step08_ViSoBERT" / "Datas" / "eval"
PHASE55_BASE = WORKSPACE / "Step05_SSOT" / "Datas" / "phase55"
OUT_BASE = WORKSPACE / "Step08_ViSoBERT" / "Datas" / "eval"


def _error_type(gold: str, pred: str) -> str:
    g, p = gold.upper(), pred.upper()
    if g == p:
        return "correct"
    if g == "NEUTRAL" and p == "POSITIVE":
        return "NEU_as_POS"
    if g == "NEUTRAL" and p == "NEGATIVE":
        return "NEU_as_NEG"
    if g == "POSITIVE" and p == "NEUTRAL":
        return "POS_as_NEU"
    if g == "NEGATIVE" and p == "NEUTRAL":
        return "NEG_as_NEU"
    if g == "POSITIVE" and p == "NEGATIVE":
        return "POS_as_NEG"
    if g == "NEGATIVE" and p == "POSITIVE":
        return "NEG_as_POS"
    return f"{g}_as_{p}"


def main() -> None:
    parser = argparse.ArgumentParser(description="T3 sentiment error analysis (misclassified cases).")
    parser.add_argument("--eval-tag", default="phase55_tier23_001_t3_human")
    parser.add_argument("--phase55-tag", default="phase55_tier23_001")
    parser.add_argument("--min-cases", type=int, default=30)
    args = parser.parse_args()

    pred_path = EVAL_BASE / args.eval_tag / "t3_predictions.csv"
    frame_path = PHASE55_BASE / args.phase55_tag / "tier3" / "tier3_mini_iaa_sampling_frame.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    preds = pd.read_csv(pred_path)
    frame = pd.read_csv(frame_path)
    frame["clause_id_final"] = frame["clause_id_final"].astype(str)
    preds["clause_id_final"] = preds["clause_id_final"].astype(str)

    merged = preds.merge(
        frame[["clause_id_final", "clause_text", "clause_text_normalized", "sentiment_polarity_hint"]],
        on="clause_id_final",
        how="left",
    )
    merged["text"] = merged["clause_text_normalized"].fillna("").astype(str)
    merged.loc[merged["text"].str.strip() == "", "text"] = merged["clause_text"].fillna("").astype(str)
    merged["error_type"] = merged.apply(
        lambda r: _error_type(str(r["label_gold"]), str(r["label_pred"])), axis=1
    )

    errors = merged.loc[~merged["agree"]].copy()
    errors = errors.sort_values(["error_type", "confidence"], ascending=[True, True])

    # Prioritize NEU boundary cases for paper discussion
    priority_types = ["NEU_as_POS", "NEU_as_NEG", "POS_as_NEU", "NEG_as_NEU", "POS_as_NEG", "NEG_as_POS"]
    errors["priority"] = errors["error_type"].apply(
        lambda t: priority_types.index(t) if t in priority_types else 99
    )
    errors = errors.sort_values(["priority", "confidence"])

    out_dir = OUT_BASE / args.eval_tag
    error_path = out_dir / "t3_error_analysis.csv"
    errors.to_csv(error_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_tag": args.eval_tag,
        "total_clauses": int(len(merged)),
        "misclassified": int(len(errors)),
        "min_cases_required": args.min_cases,
        "meets_min_cases": len(errors) >= args.min_cases,
        "error_type_counts": errors["error_type"].value_counts().to_dict(),
        "low_confidence_errors": int((errors["confidence"].astype(float) < 0.6).sum()) if "confidence" in errors.columns else 0,
        "output_file": str(error_path),
    }
    summary_path = out_dir / "t3_error_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", error_path)


if __name__ == "__main__":
    main()
