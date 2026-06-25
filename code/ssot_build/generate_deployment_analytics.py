"""
Deployment diagnostic analytics: silver pipeline labels vs model predictions.

Appendix-only; not primary F1 claims.
Outputs: Step05_SSOT/Datas/publication/<run_tag>/deployment/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
INPUT_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "downstream_inputs"
PRED_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "predictions"
OUT_BASE = WORKSPACE / "Step05_SSOT" / "Datas" / "publication"
STEP08_TOOLS = WORKSPACE / "Step08_ViSoBERT" / "Tools"

PRED_SENTIMENT_MAP = {"POS": "POS", "NEU": "NEU", "NEG": "NEG", "POSITIVE": "POS", "NEUTRAL": "NEU", "NEGATIVE": "NEG"}


def _load_pool(run_tag: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(INPUT_BASE / run_tag / f"{split}.csv")
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _load_aspect_preds(run_tag: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        path = PRED_BASE / run_tag / f"{split}_predictions.csv"
        sub = pd.read_csv(path)[["clause_id_final", "pred_aspect"]]
        sub["split"] = split
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out["clause_id_final"] = out["clause_id_final"].astype(str)
    return out.drop_duplicates("clause_id_final")


def _clause_text(row: pd.Series) -> str:
    norm = str(row.get("clause_text_normalized", "") or "").strip()
    if norm:
        return norm
    return str(row.get("clause_text", "") or "").strip()


def _run_clause_sentiment(pool: pd.DataFrame, cache_path: Path, batch_size: int, max_length: int) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["clause_id_final"] = cached["clause_id_final"].astype(str)
        return cached

    if str(STEP08_TOOLS) not in sys.path:
        sys.path.insert(0, str(STEP08_TOOLS))
    from visobert_sentiment import build_pipeline, run_inference  # noqa: WPS433

    infer_df = pool.copy()
    infer_df["clause_id_final"] = infer_df["clause_id_final"].astype(str)
    infer_df["id"] = infer_df["clause_id_final"]
    texts = infer_df.apply(_clause_text, axis=1)
    infer_df["review_text_sentence_segmented"] = texts.apply(
        lambda t: json.dumps([t], ensure_ascii=False)
    )

    pipe = build_pipeline(batch_size, max_length)
    review_data, _, _ = run_inference(
        df=infer_df,
        text_col="review_text_sentence_segmented",
        pipe=pipe,
        low_conf_thr=0.60,
        no_preprocess=False,
    )

    rows = []
    for item in review_data:
        cid = str(item["id"])
        labels = item["results"]["labels"]
        confs = item["results"]["confs"]
        raw = labels[0] if labels else "NEU"
        pred = PRED_SENTIMENT_MAP.get(str(raw).strip().upper(), "NEU")
        rows.append(
            {
                "clause_id_final": cid,
                "pred_sentiment": pred,
                "pred_sentiment_raw": raw,
                "confidence": confs[0] if confs else None,
            }
        )
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def _agreement_table(silver: pd.Series, model: pd.Series) -> dict:
    agree = (silver.astype(str) == model.astype(str))
    return {
        "n": int(len(silver)),
        "agreement_n": int(agree.sum()),
        "agreement_rate": round(float(agree.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment diagnostic: silver vs model preds.")
    parser.add_argument("--run-tag", default="phase55_final_001")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--skip-sentiment-inference", action="store_true")
    args = parser.parse_args()

    pool = _load_pool(args.run_tag)
    pool["clause_id_final"] = pool["clause_id_final"].astype(str)
    aspect_preds = _load_aspect_preds(args.run_tag)

    merged = pool.merge(aspect_preds, on="clause_id_final", how="left", validate="one_to_one")
    merged["silver_aspect"] = merged["aspect_hint"].astype(str)
    merged["silver_sentiment"] = merged["sentiment_polarity_hint"].astype(str)

    out_dir = OUT_BASE / args.run_tag / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "clause_sentiment_pred.csv"
    if args.skip_sentiment_inference and not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}; run without --skip-sentiment-inference first")

    if not args.skip_sentiment_inference:
        sent = _run_clause_sentiment(pool, cache_path, args.batch_size, args.max_length)
    else:
        sent = pd.read_csv(cache_path)
        sent["clause_id_final"] = sent["clause_id_final"].astype(str)

    merged = merged.merge(sent, on="clause_id_final", how="left", validate="one_to_one")

    m1_overall = _agreement_table(merged["silver_aspect"], merged["pred_aspect"].astype(str))
    m1_per_class = {}
    for lbl in sorted(merged["silver_aspect"].unique()):
        sub = merged.loc[merged["silver_aspect"] == lbl]
        m1_per_class[lbl] = _agreement_table(sub["silver_aspect"], sub["pred_aspect"].astype(str))

    neg_sub = merged.loc[merged["silver_sentiment"] == "NEG"]
    m2_neg = _agreement_table(neg_sub["silver_sentiment"], neg_sub["pred_sentiment"].astype(str))

    disagree = merged.loc[merged["silver_aspect"] != merged["pred_aspect"].astype(str)].copy()
    cross_disagree = (
        pd.crosstab(disagree["silver_aspect"], disagree["pred_aspect"].astype(str))
        if len(disagree)
        else pd.DataFrame()
    )

    sent_disagree = merged.loc[merged["silver_sentiment"] != merged["pred_sentiment"].astype(str)].copy()
    sent_cross = (
        pd.crosstab(
            sent_disagree["silver_aspect"],
            sent_disagree["silver_sentiment"] + "_vs_" + sent_disagree["pred_sentiment"].astype(str),
        )
        if len(sent_disagree)
        else pd.DataFrame()
    )

    merged.to_csv(out_dir / "deployment_merged_clauses.csv", index=False, encoding="utf-8-sig")
    if len(cross_disagree):
        cross_disagree.to_csv(out_dir / "aspect_disagreement_crosstab.csv", encoding="utf-8-sig")

    top_disagree = {}
    if len(cross_disagree):
        stacked = cross_disagree.stack().sort_values(ascending=False).head(10)
        top_disagree = {f"{a}|{b}": int(v) for (a, b), v in stacked.items()}

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "framing": "deployment_diagnostic_appendix_only",
        "n_clauses": int(len(merged)),
        "M1_silver_vs_model_aspect": {"overall": m1_overall, "per_silver_class": m1_per_class},
        "M2_silver_vs_model_sentiment_on_neg_subset": m2_neg,
        "M3_top_aspect_disagreement_pairs": top_disagree,
        "sentiment_disagreement_n": int(len(sent_disagree)),
        "note": "Not for primary accuracy claims; illustrates silver-label vs deployed inference gap.",
    }

    summary_path = out_dir / "deployment_analytics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
