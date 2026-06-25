"""
Step 7 primary baseline: clause-level aspect classification.
Uses TF-IDF + LogisticRegression on phase55 downstream splits.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline


WORKSPACE = Path(__file__).resolve().parents[2]
INPUT_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "downstream_inputs"
OUT_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "predictions"


def _resolve_run_tag(run_tag: str | None) -> str:
    if run_tag:
        return run_tag
    latest = INPUT_BASE / "LATEST.txt"
    if not latest.exists():
        raise FileNotFoundError("Missing Step07 downstream_inputs/LATEST.txt")
    return latest.read_text(encoding="utf-8").strip().splitlines()[0]


def _load_split(run_tag: str, split_name: str) -> pd.DataFrame:
    path = INPUT_BASE / run_tag / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def _text_series(df: pd.DataFrame) -> pd.Series:
    norm = df.get("clause_text_normalized", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    raw = df.get("clause_text", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    return norm.where(norm != "", raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step07 aspect baseline classifier.")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    run_tag = _resolve_run_tag(args.run_tag)
    train = _load_split(run_tag, "train")
    val = _load_split(run_tag, "val")
    test = _load_split(run_tag, "test")

    for name, df in (("train", train), ("val", val), ("test", test)):
        if "aspect_hint" not in df.columns:
            raise ValueError(f"{name} split missing aspect_hint")

    x_train = _text_series(train)
    y_train = train["aspect_hint"].astype(str)
    x_val = _text_series(val)
    y_val = val["aspect_hint"].astype(str)
    x_test = _text_series(test)
    y_test = test["aspect_hint"].astype(str)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    out_dir = OUT_BASE / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = train.copy()
    train_out["pred_aspect"] = train_pred
    val_out = val.copy()
    val_out["pred_aspect"] = val_pred
    test_out = test.copy()
    test_out["pred_aspect"] = test_pred
    train_out.to_csv(out_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")
    val_out.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_out.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
    }
    (out_dir / "aspect_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_BASE / "LATEST.txt").write_text(f"{run_tag}\n", encoding="utf-8")

    print(json.dumps({"run_tag": run_tag, "val_macro_f1": summary["val_macro_f1"], "test_macro_f1": summary["test_macro_f1"]}, indent=2))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
