"""
Prepare ViSoBERT-ready review-level datasets from split clause files.

Input source:
- Step08_ViSoBERT/Datas/downstream_inputs/<run_tag>/{train,val,test}.csv

Output:
- Step08_ViSoBERT/Datas/model_ready/<run_tag>/{train,val,test}_reviews.csv
- Step08_ViSoBERT/Datas/model_ready/<run_tag>/{val,test}_reviews_human_gold.csv
- Step08_ViSoBERT/Datas/model_ready/<run_tag>/summary.json

These outputs can be used directly with `visobert_sentiment.py`:
- required columns: id, review_text_sentence_segmented
- optional validation label column: label
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_DIR = WORKSPACE / "Step08_ViSoBERT" / "Datas"
INPUT_BASE = BASE_DIR / "downstream_inputs"
OUTPUT_BASE = BASE_DIR / "model_ready"

LABEL_MAP = {"POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL"}


def _resolve_run_tag(run_tag: str | None) -> str:
    if run_tag:
        return run_tag
    latest = INPUT_BASE / "LATEST.txt"
    if not latest.exists():
        raise FileNotFoundError("Cannot find downstream_inputs/LATEST.txt")
    return latest.read_text(encoding="utf-8").strip().splitlines()[0]


def _load_split_df(run_tag: str, split_name: str) -> pd.DataFrame:
    path = INPUT_BASE / run_tag / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def _load_train_union(path_arg: str | None) -> pd.DataFrame | None:
    if not path_arg:
        return None
    p = Path(path_arg)
    if not p.is_absolute():
        p = WORKSPACE / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Train union file not found: {p}")
    return pd.read_csv(p)


def _choose_clause_text(df: pd.DataFrame) -> pd.Series:
    if "clause_text_normalized" in df.columns:
        text = df["clause_text_normalized"].fillna("").astype(str).str.strip()
        if (text != "").any():
            return text
    if "clause_text" in df.columns:
        return df["clause_text"].fillna("").astype(str).str.strip()
    raise ValueError("Input does not contain clause_text_normalized or clause_text.")


def _majority_label(labels: list[str]) -> str:
    mapped = []
    for raw in labels:
        val = str(raw).strip().upper()
        if val == "MIXED":
            # Explicit policy: map rare MIXED to NEUTRAL for 3-class model.
            mapped.append("NEUTRAL")
            continue
        mapped.append(LABEL_MAP.get(val, "NEUTRAL"))

    counts = Counter(mapped)
    # Stable tie-break preference: NEGATIVE > NEUTRAL > POSITIVE
    # (conservative for risk-sensitive sentiment detection).
    for cand in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
        if counts[cand] == max(counts.values()):
            return cand
    return "NEUTRAL"


def _prepare_review_level(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    work = df.copy()
    work["review_id"] = work["review_id"].astype(str)
    work["clause_text_final"] = _choose_clause_text(work)
    work["is_human_gold_clause"] = work["annotation_source"].astype(str).str.startswith("human_gold")

    rows = []
    grouped = work.groupby("review_id", sort=False)
    for review_id, g in grouped:
        clause_texts = [t for t in g["clause_text_final"].tolist() if str(t).strip() != ""]
        if not clause_texts:
            continue

        labels = g["sentiment_polarity_hint"].astype(str).tolist() if "sentiment_polarity_hint" in g.columns else []
        label = _majority_label(labels)
        human_gold_ratio = float(g["is_human_gold_clause"].mean())

        rows.append(
            {
                "id": review_id,
                "review_id": review_id,
                "split": split_name,
                "review_text_sentence_segmented": json.dumps(clause_texts, ensure_ascii=False),
                "review_text_cleaned": " || ".join(clause_texts),
                "num_clauses": int(len(clause_texts)),
                "label": label,
                "has_human_gold_clause": bool(g["is_human_gold_clause"].any()),
                "human_gold_clause_ratio": round(human_gold_ratio, 4),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ViSoBERT model-ready review datasets.")
    parser.add_argument("--run-tag", default=None, help="Optional downstream input run tag.")
    parser.add_argument(
        "--train-union-file",
        default=None,
        help="Optional train union CSV (extended + Tier-2). Used only for train split.",
    )
    args = parser.parse_args()

    run_tag = _resolve_run_tag(args.run_tag)
    out_dir = OUTPUT_BASE / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_union_df = _load_train_union(args.train_union_file)

    prepared = {}
    for split_name in ("train", "val", "test"):
        if split_name == "train" and train_union_df is not None:
            df = train_union_df.copy()
        else:
            df = _load_split_df(run_tag, split_name)
        prepped = _prepare_review_level(df, split_name)
        prepared[split_name] = prepped
        prepped.to_csv(out_dir / f"{split_name}_reviews.csv", index=False, encoding="utf-8-sig")

    for split_name in ("val", "test"):
        hg = prepared[split_name].loc[prepared[split_name]["has_human_gold_clause"] == True].copy()
        hg.to_csv(out_dir / f"{split_name}_reviews_human_gold.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviews_by_split": {k: int(len(v)) for k, v in prepared.items()},
        "label_distribution": {
            k: v["label"].value_counts().to_dict() for k, v in prepared.items()
        },
        "human_gold_reviews_by_split": {
            k: int(prepared[k]["has_human_gold_clause"].sum()) for k in ("val", "test")
        },
        "policy": {
            "source_label": "sentiment_polarity_hint (clause-level) -> majority vote at review-level",
            "mixed_handling": "MIXED -> NEUTRAL",
            "tie_break": "NEGATIVE > NEUTRAL > POSITIVE",
            "train_source": "train_union_file" if train_union_df is not None else "downstream_inputs/train.csv",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_BASE / "LATEST.txt").write_text(f"{run_tag}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
