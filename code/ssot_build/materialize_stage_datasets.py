"""
Materialize full-feature train/val/test CSVs for downstream stages.

Stages:
- Step06_BERTopic
- Step07_Aspect
- Step08_ViSoBERT

Input contract:
- SSOT final dataset from Step05_SSOT/Datas/clauses_ssot/clauses_final_43c_44_*.csv
- Split file from Step05_SSOT/Datas/clauses_ssot/splits/downstream_split_reviewid_*.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
SSOT_DIR = WORKSPACE / "Step05_SSOT" / "Datas" / "clauses_ssot"
SPLIT_DIR = SSOT_DIR / "splits"

STAGE_OUTPUT_DIRS = {
    "Step06_BERTopic": WORKSPACE / "Step06_BERTopic" / "Datas" / "downstream_inputs",
    "Step07_Aspect": WORKSPACE / "Step07_Aspect" / "Datas" / "downstream_inputs",
    "Step08_ViSoBERT": WORKSPACE / "Step08_ViSoBERT" / "Datas" / "downstream_inputs",
}


def _resolve_ssot(input_arg: str | None) -> Path:
    if input_arg:
        p = Path(input_arg)
        if not p.is_absolute():
            p = WORKSPACE / p
        return p.resolve()

    latest_file = SSOT_DIR / "LATEST.txt"
    if latest_file.exists():
        first_line = latest_file.read_text(encoding="utf-8").strip().splitlines()[0]
        candidate = SSOT_DIR / first_line
        if candidate.exists():
            return candidate

    candidates = sorted(SSOT_DIR.glob("clauses_final_43c_44_*.csv"))
    if not candidates:
        raise FileNotFoundError("No SSOT final dataset found.")
    return candidates[-1]


def _resolve_split(split_arg: str | None) -> Path:
    if split_arg:
        p = Path(split_arg)
        if not p.is_absolute():
            p = WORKSPACE / p
        return p.resolve()

    latest_file = SPLIT_DIR / "LATEST_SPLIT.txt"
    if latest_file.exists():
        first_line = latest_file.read_text(encoding="utf-8").strip().splitlines()[0]
        candidate = SPLIT_DIR / first_line
        if candidate.exists():
            return candidate

    candidates = sorted(SPLIT_DIR.glob("downstream_split_reviewid_*.csv"))
    if not candidates:
        raise FileNotFoundError("No downstream split file found.")
    return candidates[-1]


def _assert_split_integrity(df: pd.DataFrame) -> None:
    required_cols = {"clause_id_final", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")

    allowed = {"train", "val", "test"}
    bad_values = sorted(set(df["split"].dropna().astype(str)) - allowed)
    if bad_values:
        raise ValueError(f"Unexpected split values: {bad_values}")

    if df["clause_id_final"].duplicated().any():
        dup_count = int(df["clause_id_final"].duplicated().sum())
        raise ValueError(f"Split file has duplicate clause_id_final rows: {dup_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize stage train/val/test full-feature CSVs.")
    parser.add_argument("--input", type=str, default=None, help="Optional SSOT clauses_final path.")
    parser.add_argument("--split-file", type=str, default=None, help="Optional split CSV path.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag override.")
    args = parser.parse_args()

    ssot_path = _resolve_ssot(args.input)
    split_path = _resolve_split(args.split_file)
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    ssot_df = pd.read_csv(ssot_path)
    split_df = pd.read_csv(split_path)
    _assert_split_integrity(split_df)

    if "clause_id_final" not in ssot_df.columns:
        raise ValueError("SSOT file missing clause_id_final.")
    if "has_extended_flags" not in ssot_df.columns:
        raise ValueError("SSOT file missing has_extended_flags.")

    scope_df = ssot_df.loc[ssot_df["has_extended_flags"] == True].copy()
    merged = scope_df.merge(
        split_df[["clause_id_final", "split"]],
        on="clause_id_final",
        how="inner",
        validate="one_to_one",
    )

    missing_from_split = set(scope_df["clause_id_final"]) - set(merged["clause_id_final"])
    if missing_from_split:
        raise RuntimeError(
            f"{len(missing_from_split)} extended clauses do not have split assignment."
        )

    summary = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_ssot": str(ssot_path),
        "input_split": str(split_path),
        "rows_scope": int(len(merged)),
        "rows_by_split": {k: int(v) for k, v in merged["split"].value_counts().to_dict().items()},
        "reviews_by_split": {
            k: int(v)
            for k, v in merged.groupby("split")["review_id"].nunique().to_dict().items()
        },
    }

    for stage_name, base_dir in STAGE_OUTPUT_DIRS.items():
        run_dir = base_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "val", "test"):
            split_out = merged.loc[merged["split"] == split_name].copy()
            out_path = run_dir / f"{split_name}.csv"
            split_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        manifest = {
            **summary,
            "stage": stage_name,
            "output_dir": str(run_dir),
            "files": {
                "train": str((run_dir / "train.csv").name),
                "val": str((run_dir / "val.csv").name),
                "test": str((run_dir / "test.csv").name),
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (base_dir / "LATEST.txt").write_text(f"{run_tag}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for stage_name, base_dir in STAGE_OUTPUT_DIRS.items():
        print(f"Wrote {stage_name}: {base_dir / run_tag}")


if __name__ == "__main__":
    main()
