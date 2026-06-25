"""
Prepare BERTopic-ready inputs from downstream split files.

Input source:
- Step06_BERTopic/Datas/downstream_inputs/<run_tag>/{train,val,test}.csv

Output:
- Step06_BERTopic/Datas/model_ready/<run_tag>/bertopic_{split}_clauses.csv
- Step06_BERTopic/Datas/model_ready/<run_tag>/bertopic_all_clauses.csv
- Step06_BERTopic/Datas/model_ready/<run_tag>/summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_DIR = WORKSPACE / "Step06_BERTopic" / "Datas"
INPUT_BASE = BASE_DIR / "downstream_inputs"
OUTPUT_BASE = BASE_DIR / "model_ready"


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


def _pick_text(df: pd.DataFrame) -> pd.Series:
    norm = None
    raw = None
    if "clause_text_normalized" in df.columns:
        norm = df["clause_text_normalized"].fillna("").astype(str).str.strip()
    if "clause_text" in df.columns:
        raw = df["clause_text"].fillna("").astype(str).str.strip()

    # Row-wise fallback: prefer normalized text, fallback to raw clause text
    # when normalized is empty. This avoids dropping valid clauses.
    if norm is not None and raw is not None:
        return norm.where(norm != "", raw)
    if norm is not None:
        return norm
    if raw is not None:
        return raw
    raise ValueError("Input does not contain clause_text_normalized or clause_text.")


def _prepare_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    work = df.copy()
    work["doc_id"] = work["clause_id_final"].astype(str)
    work["review_id"] = work["review_id"].astype(str)
    work["text"] = _pick_text(work)
    work = work.sort_values(["review_id", "doc_id"]).reset_index(drop=True)

    # Context fusion required by v3.1: [prev ; target ; next] in same review.
    work["prev_text"] = work.groupby("review_id")["text"].shift(1).fillna("")
    work["next_text"] = work.groupby("review_id")["text"].shift(-1).fillna("")
    work["context_fused_text"] = (
        work["prev_text"].astype(str).str.strip()
        + " [SEP] "
        + work["text"].astype(str).str.strip()
        + " [SEP] "
        + work["next_text"].astype(str).str.strip()
    ).str.strip()

    out = pd.DataFrame()
    out["doc_id"] = work["doc_id"]
    out["review_id"] = work["review_id"]
    out["text"] = work["text"]
    out["context_fused_text"] = work["context_fused_text"]
    out["split"] = split_name

    for col in ["aspect_hint", "sentiment_polarity_hint", "annotation_source", "contains_green_signal"]:
        if col in work.columns:
            out[col] = work[col]

    out = out.loc[out["text"] != ""].copy()
    out = out.drop_duplicates(subset=["doc_id"], keep="first")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BERTopic model-ready inputs.")
    parser.add_argument("--run-tag", default=None, help="Optional downstream input run tag.")
    args = parser.parse_args()

    run_tag = _resolve_run_tag(args.run_tag)
    out_dir = OUTPUT_BASE / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for split_name in ("train", "val", "test"):
        df = _load_split_df(run_tag, split_name)
        prepped = _prepare_split(df, split_name)
        prepared[split_name] = prepped
        prepped.to_csv(out_dir / f"bertopic_{split_name}_clauses.csv", index=False, encoding="utf-8-sig")
        prepped.to_csv(out_dir / f"bertopic_{split_name}_context_fused.csv", index=False, encoding="utf-8-sig")

    all_df = pd.concat([prepared["train"], prepared["val"], prepared["test"]], ignore_index=True)
    all_df.to_csv(out_dir / "bertopic_all_clauses.csv", index=False, encoding="utf-8-sig")
    all_df.to_csv(out_dir / "bertopic_all_context_fused.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_by_split": {k: int(len(v)) for k, v in prepared.items()},
        "rows_total": int(len(all_df)),
        "unique_reviews_total": int(all_df["review_id"].nunique()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_BASE / "LATEST.txt").write_text(f"{run_tag}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
