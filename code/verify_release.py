"""Release hygiene checks for the public reproducibility package.

The script checks for common publication blockers:
- direct identifiers in released text fields;
- absolute local workspace paths in released metadata;
- accidental secret tokens;
- expected row counts for the main public datasets.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LOCAL_PATH_RE = re.compile(
    r"(?:C:/Users|C:\\Users|E:\\[A-Za-z0-9_. -]|G:\\[A-Za-z0-9_. -]|"
    r"OneDrive|Other computers|DangXuLy|BanThao|zBackup)"
)
SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{20,}|"
    r"OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|GITHUB_TOKEN|Bearer\s+[A-Za-z0-9._\-]+)",
    re.IGNORECASE,
)

EXPECTED_ROWS = {
    Path("data/processed/clauses_final_43c_44_20260528_034911.csv"): 4254,
    Path("data/splits/downstream_split_reviewid_20260528_022322.csv"): 1724,
    Path("outputs/evaluation/aspect_t2_gold_holdout/t2_predictions.csv"): 297,
    Path("outputs/evaluation/sentiment_t3_primary_no_adrd25/t3_predictions.csv"): 117,
}


def iter_text_files() -> list[Path]:
    suffixes = {".csv", ".json", ".md", ".txt", ".py", ".ipynb", ".cff", ".gitignore"}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and (path.suffix.lower() in suffixes or path.name == "LICENSE"):
            files.append(path)
    return files


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def check_expected_rows(errors: list[str]) -> None:
    for rel_path, expected in EXPECTED_ROWS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing expected file: {rel_path.as_posix()}")
            continue
        actual = count_csv_rows(path)
        if actual != expected:
            errors.append(f"Unexpected row count for {rel_path.as_posix()}: {actual} != {expected}")


def check_text_fields(errors: list[str]) -> None:
    for path in ROOT.rglob("*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            text_columns = [
                col
                for col in (reader.fieldnames or [])
                if "text" in col.lower() or "sentence" in col.lower()
            ]
            if not text_columns:
                continue
            for row_number, row in enumerate(reader, start=2):
                for col in text_columns:
                    value = row.get(col, "") or ""
                    if PHONE_RE.search(value):
                        rel = path.relative_to(ROOT).as_posix()
                        errors.append(f"Phone-like string in {rel}:{row_number}:{col}")
                    if EMAIL_RE.search(value):
                        rel = path.relative_to(ROOT).as_posix()
                        errors.append(f"Email-like string in {rel}:{row_number}:{col}")


def check_global_patterns(errors: list[str]) -> None:
    for path in iter_text_files():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8-sig", errors="replace")

        if rel == "code/verify_release.py":
            continue

        if SECRET_RE.search(text):
            errors.append(f"Secret-like token in {rel}")

        # URLs in README/CITATION are expected; local absolute paths are not.
        if LOCAL_PATH_RE.search(text):
            errors.append(f"Local workspace path marker in {rel}")


def check_json_valid(errors: list[str]) -> None:
    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            rel = path.relative_to(ROOT).as_posix()
            errors.append(f"Invalid JSON in {rel}: {exc}")


def main() -> int:
    errors: list[str] = []
    check_expected_rows(errors)
    check_text_fields(errors)
    check_global_patterns(errors)
    check_json_valid(errors)

    if errors:
        print("RELEASE CHECK FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print("RELEASE CHECK PASSED")
    print(f"Repository: {ROOT}")
    print(f"Checked files: {len(iter_text_files())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
