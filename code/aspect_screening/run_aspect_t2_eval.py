"""
Step 7 primary evaluation on T2 human gold (n=297 aspect IAA sample).

Trains TF-IDF + LogisticRegression on silver train split, evaluates on
Step G gold clauses (gold_aspect_hint) — primary aspect claim lane per v3 §7.2.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


WORKSPACE = Path(__file__).resolve().parents[2]
INPUT_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "downstream_inputs"
GOLD_PATH = (
    WORKSPACE
    / "Step04_41_43c"
    / "Datas"
    / "43c_llm_assisted"
    / "stepG_human_review"
    / "iaa_batches"
    / "stepG_gold_labels_v1.csv"
)
OUT_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "eval"


def _resolve_run_tag(run_tag: str | None) -> str:
    if run_tag:
        return run_tag
    latest = INPUT_BASE / "LATEST.txt"
    if not latest.exists():
        raise FileNotFoundError("Missing Step07 downstream_inputs/LATEST.txt")
    return latest.read_text(encoding="utf-8").strip().splitlines()[0]


def _text_series(df: pd.DataFrame) -> pd.Series:
    norm = df.get("clause_text_normalized", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    raw = df.get("clause_text", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    return norm.where(norm != "", raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7 T2 gold eval (n=297 human aspect labels).")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Exclude T2 gold clause_ids from silver train fit (conservative audit protocol).",
    )
    args = parser.parse_args()

    run_tag = _resolve_run_tag(args.run_tag)
    if args.output_tag:
        output_tag = args.output_tag
    elif args.holdout:
        output_tag = f"{run_tag}_t2_gold_holdout"
    else:
        output_tag = f"{run_tag}_t2_gold"

    train = pd.read_csv(INPUT_BASE / run_tag / "train.csv")
    train["clause_id_final"] = train["clause_id_final"].astype(str)
    gold = pd.read_csv(GOLD_PATH)
    gold["clause_id_final"] = gold["clause_id_final"].astype(str)

    # Join gold text from gold file (clause_text column present)
    t2 = gold[["clause_id_final", "clause_text", "gold_aspect_hint"]].copy()
    t2 = t2.rename(columns={"gold_aspect_hint": "label_gold"})
    t2 = t2.loc[t2["label_gold"].astype(str).str.strip() != ""].copy()
    if len(t2) != 297:
        raise RuntimeError(f"Expected 297 T2 gold rows, got {len(t2)}")

    t2_ids = set(t2["clause_id_final"].astype(str))
    n_overlap_before = int(train["clause_id_final"].isin(t2_ids).sum())
    if args.holdout:
        train_fit = train.loc[~train["clause_id_final"].isin(t2_ids)].copy()
    else:
        train_fit = train.copy()

    x_train = _text_series(train_fit)
    y_train = train_fit["aspect_hint"].astype(str)
    x_t2 = t2["clause_text"].fillna("").astype(str)
    y_t2 = t2["label_gold"].astype(str)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_t2)

    labels = sorted(set(y_t2) | set(y_pred))
    macro_f1 = float(f1_score(y_t2, y_pred, average="macro", zero_division=0, labels=labels))
    report = classification_report(y_t2, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_t2, y_pred, labels=labels)
    cm_dict = {labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))} for i in range(len(labels))}

    per_class_f1 = {
        lbl: {
            "f1": round(report.get(lbl, {}).get("f1-score", 0.0), 4),
            "support": int(report.get(lbl, {}).get("support", 0)),
        }
        for lbl in labels
        if lbl in report
    }

    out_dir = OUT_BASE / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = t2.copy()
    pred_df["label_pred"] = y_pred
    pred_df["agree"] = pred_df["label_gold"].astype(str) == pred_df["label_pred"].astype(str)
    pred_df.to_csv(out_dir / "t2_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_type": "step7_primary_t2_gold_holdout" if args.holdout else "step7_primary_t2_gold",
        "run_tag": run_tag,
        "output_tag": output_tag,
        "model": "TF-IDF + LogisticRegression (silver train)",
        "holdout_t2_from_train": bool(args.holdout),
        "n_train_rows_before_holdout": int(len(train)),
        "n_train_rows_after_holdout": int(len(train_fit)),
        "n_t2_overlap_in_train_before_holdout": n_overlap_before,
        "n_t2_gold": int(len(t2)),
        "macro_f1": round(macro_f1, 4),
        "agreement_with_gold": int(pred_df["agree"].sum()),
        "per_class": per_class_f1,
        "confusion_matrix": cm_dict,
        "gold_source": str(GOLD_PATH),
        "note": (
            "Primary aspect claim lane per v3 §7.2; T2 gold clauses excluded from train fit."
            if args.holdout
            else "Primary aspect claim lane per v3 §7.2; not T1 silver test (legacy: T2 may overlap train)."
        ),
    }

    report_name = "aspect_eval_T2_holdout.json" if args.holdout else "aspect_eval_T2.json"
    report_path = out_dir / report_name
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_BASE / "LATEST.txt").write_text(f"{output_tag}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", report_path)


if __name__ == "__main__":
    main()
