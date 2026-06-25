"""
Build leakage-safe downstream splits grouped by review_id.

Default behavior:
- input: latest Step H SSOT dataset from Step05_SSOT/Datas/clauses_ssot
- scope: has_extended_flags == True (downstream 6-9)
- split: train/val/test by review_id (no review leaks across splits)
- balancing: keep human-gold review ratio close to global ratio in each split
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
SSOT_DIR = WORKSPACE / "Step05_SSOT" / "Datas" / "clauses_ssot"
OUT_DIR = SSOT_DIR / "splits"


def _resolve_input(path_arg: str | None) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.is_absolute():
            p = WORKSPACE / p
        return p.resolve()

    latest = SSOT_DIR / "LATEST.txt"
    if latest.exists():
        first_line = latest.read_text(encoding="utf-8").strip().splitlines()[0]
        candidate = SSOT_DIR / first_line
        if candidate.exists():
            return candidate

    candidates = sorted(SSOT_DIR.glob("clauses_final_43c_44_*.csv"))
    if not candidates:
        raise FileNotFoundError("No SSOT final dataset found in clauses_ssot.")
    return candidates[-1]


def _assign_review_splits(
    reviews: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    reviews = reviews.copy()
    n_reviews = len(reviews)
    if n_reviews < 3:
        raise ValueError("Need at least 3 reviews to create train/val/test splits.")

    target_test = int(round(n_reviews * test_ratio))
    target_val = int(round(n_reviews * val_ratio))
    target_train = n_reviews - target_test - target_val

    # Keep at least one review in each split.
    if min(target_train, target_val, target_test) <= 0:
        raise ValueError(
            f"Invalid split counts (train={target_train}, val={target_val}, test={target_test}). "
            "Adjust ratios for current dataset size."
        )

    human_reviews = reviews.loc[reviews["has_human_gold"], "review_id"].tolist()
    non_human_reviews = reviews.loc[~reviews["has_human_gold"], "review_id"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(human_reviews)
    rng.shuffle(non_human_reviews)

    human_ratio = len(human_reviews) / n_reviews
    target_h_train = min(len(human_reviews), int(round(target_train * human_ratio)))
    target_h_val = min(len(human_reviews) - target_h_train, int(round(target_val * human_ratio)))
    target_h_test = min(
        len(human_reviews) - target_h_train - target_h_val,
        int(round(target_test * human_ratio)),
    )

    # Distribute any remaining human reviews to train first (better supervision),
    # then val, then test.
    remaining_h = len(human_reviews) - (target_h_train + target_h_val + target_h_test)
    while remaining_h > 0:
        if target_h_train < target_train:
            target_h_train += 1
        elif target_h_val < target_val:
            target_h_val += 1
        elif target_h_test < target_test:
            target_h_test += 1
        remaining_h -= 1

    train_reviews = human_reviews[:target_h_train]
    idx = target_h_train
    val_reviews = human_reviews[idx : idx + target_h_val]
    idx += target_h_val
    test_reviews = human_reviews[idx : idx + target_h_test]

    need_train = target_train - len(train_reviews)
    need_val = target_val - len(val_reviews)
    need_test = target_test - len(test_reviews)

    train_reviews += non_human_reviews[:need_train]
    idx_n = need_train
    val_reviews += non_human_reviews[idx_n : idx_n + need_val]
    idx_n += need_val
    test_reviews += non_human_reviews[idx_n : idx_n + need_test]
    idx_n += need_test

    # Any leftovers (due to rounding) go to train.
    leftovers = non_human_reviews[idx_n:]
    train_reviews += leftovers

    split_map: dict[str, str] = {}
    for rid in train_reviews:
        split_map[str(rid)] = "train"
    for rid in val_reviews:
        split_map[str(rid)] = "val"
    for rid in test_reviews:
        split_map[str(rid)] = "test"

    if len(split_map) != n_reviews:
        raise RuntimeError("Split assignment does not cover all review_id values.")
    return split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build downstream grouped splits from SSOT.")
    parser.add_argument("--input", type=str, default=None, help="Optional path to clauses_final CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    input_path = _resolve_input(args.input)
    df = pd.read_csv(input_path)

    if "has_extended_flags" not in df.columns:
        raise ValueError("Missing has_extended_flags in input dataset.")
    if "review_id" not in df.columns:
        raise ValueError("Missing review_id in input dataset.")

    scope = df.loc[df["has_extended_flags"] == True].copy()
    if scope.empty:
        raise ValueError("No rows with has_extended_flags=True.")

    scope["is_human_gold"] = scope["annotation_source"].astype(str).str.startswith("human_gold")
    scope["review_id"] = scope["review_id"].astype(str)

    review_table = (
        scope.groupby("review_id", as_index=False)
        .agg(
            n_clauses=("clause_id_final", "count"),
            has_human_gold=("is_human_gold", "max"),
        )
        .sort_values("review_id")
    )

    split_map = _assign_review_splits(
        reviews=review_table,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    scope["split"] = scope["review_id"].map(split_map)
    if scope["split"].isna().any():
        raise RuntimeError("Some rows are missing split assignment.")

    # Safety check: each review_id appears in exactly one split.
    review_split_nunique = scope.groupby("review_id")["split"].nunique()
    if (review_split_nunique > 1).any():
        raise RuntimeError("Leakage detected: review_id appears in multiple splits.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_file = OUT_DIR / f"downstream_split_reviewid_{run_id}.csv"
    summary_file = OUT_DIR / f"downstream_split_summary_{run_id}.json"
    latest_file = OUT_DIR / "LATEST_SPLIT.txt"

    out_cols = [
        "clause_id_final",
        "review_id",
        "split",
        "annotation_source",
        "is_human_gold",
        "aspect_hint",
        "contains_green_signal",
    ]
    out_cols = [c for c in out_cols if c in scope.columns]
    scope[out_cols].to_csv(split_file, index=False, encoding="utf-8-sig")

    split_counts = scope["split"].value_counts().to_dict()
    split_review_counts = scope.groupby("split")["review_id"].nunique().to_dict()
    split_human_counts = scope.loc[scope["is_human_gold"]].groupby("split").size().to_dict()

    summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": str(input_path),
        "scope_filter": "has_extended_flags == True",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "rows_total_scope": int(len(scope)),
        "reviews_total_scope": int(scope["review_id"].nunique()),
        "rows_by_split": {k: int(v) for k, v in split_counts.items()},
        "reviews_by_split": {k: int(v) for k, v in split_review_counts.items()},
        "human_gold_rows_by_split": {k: int(v) for k, v in split_human_counts.items()},
        "human_gold_rows_total": int(scope["is_human_gold"].sum()),
    }

    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_file.write_text(f"{split_file.name}\n{summary_file.name}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", split_file)
    print("Wrote", summary_file)


if __name__ == "__main__":
    main()
