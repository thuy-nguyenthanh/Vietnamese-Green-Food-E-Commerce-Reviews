import argparse
import ast
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

MODEL_NAME = "5CD-AI/Vietnamese-Sentiment-visobert"
DEVICE = 0 if torch.cuda.is_available() else -1

LABEL_NORMALIZE = {
    "POS": "POS",
    "LABEL_1": "POS",
    "NEU": "NEU",
    "LABEL_2": "NEU",
    "NEG": "NEG",
    "LABEL_0": "NEG",
}
SCORE_MAP = {"POS": 1, "NEU": 0, "NEG": -1}
REPORT_DEFAULT = "visobert_eval_report.json"
PARSE_LOG_DEFAULT = "visobert_parse_errors.csv"
LOW_CONF_DEFAULT = "visobert_low_confidence.csv"

TEENCODE_MAP = {
    "ko": "khong",
    "k": "khong",
    "kh": "khong",
    "khum": "khong",
    "dc": "duoc",
    "đc": "duoc",
    "sp": "san pham",
    "shoppe": "shopee",
    "shiper": "shipper",
    "mn": "mo nguoi",
    "mng": "mo nguoi",
    "oke": "ok",
    "okela": "ok",
    "chx": "chua",
    "cx": "cung",
    "mik": "minh",
    "mk": "minh",
    "vs": "voi",
    "nhaaa": "nha",
    "z": "vay",
}

MISSPELLING_MAP = {
    "nhuug": "nhưng",
    "nhưug": "nhưng",
    "giao_hành": "giao_hang",
    "giai_hàng": "giao_hang",
    "giai_hang": "giao_hang",
    "san_pham": "sản_phẩm",
    "chat_luong": "chất_lượng",
    "huong_vi": "hương_vị",
    "giao_hang": "giao hàng",
    "san pham": "sản phẩm",
    "chat luong": "chất lượng",
    "huong vi": "hương vị",
    "dong goi": "đóng gói",
    "nhiet tinh": "nhiệt tình",
    "than thien": "thân thiện",
    "hai long": "hài lòng",
    "uy tin": "uy tín",
    "shipper": "shipper",
    "siper": "shipper",
}

# Cụm cảm xúc / thái độ (sau normalize_text — khoảng trắng giữa từ)
NEGATED_ANGER_OR_FRUSTRATION = re.compile(
    r"\b(không|chưa|chẳng|chang)\b(?:\s+hề)?\s+"
    r"(tức\s+giận|bực\s+mình|bực|tức|giận|khó\s+chịu|phàn\s+nàn|cáu)\b",
    re.IGNORECASE,
)
PRAISE_ATTITUDE_PHRASE = re.compile(
    r"\b(thân\s+thiện|nhiệt\s+tình|vui\s+vẻ|chu\s+đáo|chuyên\s+nghiệp|"
    r"tận\s+tình|đáng\s+yêu|dễ\s+thương|hài\s+lòng|ưng\s+ý|tuyệt\s+vời|xuất\s+sắc|quá\s+đỉnh)\b",
    re.IGNORECASE,
)
QUANTITY_POSITIVE_PHRASE = re.compile(
    r"\b(chưa\s+(thấy\s+)?hết|ăn\s+(hoài|mãi)|dùng\s+(hoài|mãi)|nhiều\s+quá|quá\s+nhiều)\b",
    re.IGNORECASE,
)
NEUTRAL_OBJECTIVE_PHRASE = re.compile(
    r"\b(cán\s+vỡ|cán\s+dẹt|không\s+vị|hạn\s+sử\s+dụng|nguyên\s+cám|hạn\s+đến|date|nsx|hsd)\b",
    re.IGNORECASE,
)
MA_VAN_PATTERN = re.compile(r"\bmà\s+vẫn\b", re.IGNORECASE)

NEGATORS = {"khong", "chua", "chang", "chẳng", "không", "chưa"}
INTENSIFIERS = {"rat", "rất", "qua", "quá", "hoi", "hơi", "cuc_ky", "cực_kỳ"}
POSITIVE_CUES = {
    "tot",
    "tốt",
    "ngon",
    "dep",
    "đẹp",
    "hai_long",
    "hài_lòng",
    "ok",
    "uy_tin",
    "uy_tín",
    "thiện",
    "thơm",
    "rẻ",
    "nhanh",
    "ưng",
    "ổn",
    "sạch",
    "chuẩn",
    "xịn",
    "tiện",
    "tuyệt",
    "đỉnh",
}
NEGATIVE_CUES = {
    "te",
    "tệ",
    "do",
    "dở",
    "chan",
    "chán",
    "that_vong",
    "thất_vọng",
    "loi",
    "lỗi",
    "giận",
    "tức",
    "bực",
    "hôi",
    "mốc",
    "chua",
    "đắng",
    "rách",
    "lủng",
    "bể",
    "hư",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisoBERT sentiment pipeline with high-accuracy post-processing.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--low-conf-thr", type=float, default=0.60, help="Low-confidence threshold")
    parser.add_argument("--val-file", default="", help="Optional validation CSV path")
    parser.add_argument("--val-label-col", default="label", help="Gold label column in validation CSV")
    parser.add_argument("--report-file", default=REPORT_DEFAULT, help="Evaluation report JSON output")
    parser.add_argument("--parse-log-file", default=PARSE_LOG_DEFAULT, help="Parse error CSV output")
    parser.add_argument("--low-conf-file", default=LOW_CONF_DEFAULT, help="Low-confidence CSV output")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số dòng cần chạy để thử nghiệm")
    parser.add_argument(
        "--text-col",
        default="review_text_sentence_segmented",
        help="Cột chứa chuỗi list mệnh đề (sau spellcheck có thể là review_text_cleaned)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Bỏ qua các bước normalize_text, chỉ split text theo || (dùng cho cột đã qua làm sạch như clean_visobert)",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Chuẩn hóa Unicode tiếng Việt về NFC để tránh lỗi dấu tách rời.
    t = unicodedata.normalize("NFC", text).strip().lower().replace("_", " ")
    # Đồng nhất khoảng trắng và bỏ ký tự dư.
    t = re.sub(r"[“”\"'`]", " ", t)
    t = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s\-\.,!?/:;]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""

    # Sửa lỗi gõ / chính tả cụm thường gặp (trước khi tách token)
    t = re.sub(r"\bnhưug\b", "nhưng", t)
    t = re.sub(r"\bnhuug\b", "nhưng", t)
    t = re.sub(r"\bthân\s+thiệt\b", "thân thiện", t)
    t = re.sub(r"\bthan\s+thiet\b", "thân thiện", t)

    tokens = []
    for token in t.split():
        base = TEENCODE_MAP.get(token, token)
        # reduce long repeated chars: ngonnnn -> ngon
        base = re.sub(r"(.)\1{2,}", r"\1", base)
        base = MISSPELLING_MAP.get(base, base)
        # Chuẩn hóa phủ định để rule phía sau ổn định hơn.
        if base in {"khong", "k", "ko"}:
            base = "không"
        elif base == "chua":
            base = "chưa"
        tokens.append(base)

    normalized = " ".join(tokens)
    # Chuẩn hóa một số cụm nhiều từ sau khi tách token.
    for src, tgt in MISSPELLING_MAP.items():
        normalized = re.sub(rf"\b{re.escape(src)}\b", tgt, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def clean_and_parse_segments(raw_text: str, no_preprocess: bool = False) -> tuple[list[str], str | None]:
    if not isinstance(raw_text, str):
        return [], "non_string_input"

    raw = raw_text.strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return [], None

    def process_segment(s):
        s_str = str(s).strip()
        if no_preprocess:
            return s_str
        return normalize_text(s_str)

    segments: list[str] = []
    parse_error = None
    try:
        if raw.startswith("[") and raw.endswith("]"):
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                for s in result:
                    cleaned = process_segment(s)
                    if cleaned:
                        segments.append(cleaned)
            else:
                cleaned = process_segment(result)
                if cleaned:
                    segments = [cleaned]
        elif "||" in raw:
            for s in raw.split("||"):
                cleaned = process_segment(s)
                if cleaned:
                    segments.append(cleaned)
        else:
            cleaned = process_segment(raw)
            if cleaned:
                segments = [cleaned]
    except Exception as exc:
        parse_error = str(exc)
        if "||" in raw:
            for s in raw.split("||"):
                cleaned = process_segment(s)
                if cleaned:
                    segments.append(cleaned)
        else:
            cleaned = process_segment(raw)
            if cleaned:
                segments = [cleaned]
    return segments, parse_error


def adjust_score_with_rules(text: str, base_score: int) -> float:
    tokens = text.split()
    if not tokens:
        return float(base_score)

    adjusted = float(base_score)
    
    # 1. Phát hiện khối lượng nhiều/tiết kiệm (thường bị nhầm là phàn nàn/Negative)
    if QUANTITY_POSITIVE_PHRASE.search(text):
        return 1.0  # Chỉnh thẳng lên POSITIVE luôn vì đây là khen số lượng nhiều
        
    has_pos = any(tok in POSITIVE_CUES for tok in tokens)
    has_neg = any(tok in NEGATIVE_CUES for tok in tokens)
        
    # 2. Khử nhiễu cho các thuộc tính vật lý khách quan hay bị AI chấm NEG
    # Ví dụ: "cán vỡ", "cán dẹt", "không vị", "hạn sử dụng"
    if NEUTRAL_OBJECTIVE_PHRASE.search(text):
        # Nếu AI chấm âm nhưng câu này không có từ tiêu cực nặng như mốc, hôi, dở... thì kéo về Trung tính
        if base_score < 0 and not has_neg:
            return 0.0

    negated_emotion = bool(NEGATED_ANGER_OR_FRUSTRATION.search(text))
    praise_attitude = bool(PRAISE_ATTITUDE_PHRASE.search(text))
    ma_van = bool(MA_VAN_PATTERN.search(text))

    if negated_emotion and (praise_attitude or ma_van):
        adjusted = max(adjusted, 0.92)
    elif negated_emotion:
        adjusted += 0.62

    has_negator = any(tok in NEGATORS for tok in tokens)
    has_intensifier = any(tok in INTENSIFIERS for tok in tokens)

    if has_negator and has_pos and not negated_emotion:
        adjusted -= 1.0
    if has_negator and has_neg and not negated_emotion:
        adjusted += 0.5
    if has_intensifier:
        adjusted = adjusted * 1.25

    return max(-1.0, min(1.0, adjusted))


def label_from_score(value: float) -> str:
    if value > 0.33:
        return "POSITIVE"
    if value < -0.33:
        return "NEGATIVE"
    return "NEUTRAL"


def compute_review_features(
    clause_scores: list[float],
    clause_confs: list[float],
    clause_probs: list[dict[str, float]],
) -> dict:
    n = len(clause_scores)
    if n == 0:
        return {
            "num_sentences": 0,
            "num_pos": 0,
            "num_neg": 0,
            "num_neu": 0,
            "sum_score": 0.0,
            "sentiment_mean_weighted": 0.0,
            "avg_confidence": 0.0,
            "has_mixed": 0,
            "prob_neg": 0.0,
            "prob_neu": 0.0,
            "prob_pos": 0.0,
        }

    labels = [label_from_score(s) for s in clause_scores]
    num_pos = labels.count("POSITIVE")
    num_neg = labels.count("NEGATIVE")
    num_neu = labels.count("NEUTRAL")

    weighted_den = sum(max(c, 1e-6) for c in clause_confs)
    weighted_mean = sum(s * max(c, 1e-6) for s, c in zip(clause_scores, clause_confs)) / weighted_den

    prob_neg = sum(p.get("NEG", 0.0) for p in clause_probs) / n
    prob_neu = sum(p.get("NEU", 0.0) for p in clause_probs) / n
    prob_pos = sum(p.get("POS", 0.0) for p in clause_probs) / n

    return {
        "num_sentences": n,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "num_neu": num_neu,
        "sum_score": round(sum(clause_scores), 4),
        "sentiment_mean_weighted": round(weighted_mean, 4),
        "avg_confidence": round(sum(clause_confs) / n, 4),
        "has_mixed": 1 if (num_pos > 0 and num_neg > 0) else 0,
        "prob_neg": round(prob_neg, 4),
        "prob_neu": round(prob_neu, 4),
        "prob_pos": round(prob_pos, 4),
    }


def build_pipeline(batch_size: int, max_length: int):
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=DEVICE,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
        return_all_scores=True,
    )


def normalize_label(value: str) -> str:
    raw = str(value).strip().upper()
    mapping = {
        "POSITIVE": "POSITIVE",
        "POS": "POSITIVE",
        "1": "POSITIVE",
        "NEUTRAL": "NEUTRAL",
        "NEU": "NEUTRAL",
        "0": "NEUTRAL",
        "NEGATIVE": "NEGATIVE",
        "NEG": "NEGATIVE",
        "-1": "NEGATIVE",
    }
    return mapping.get(raw, "")


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict]:
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    per_class = {}
    for cls in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    macro = sum(per_class[c]["f1"] for c in labels) / len(labels)
    return round(macro, 4), per_class


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def optimize_thresholds(rows: list[dict], label_col: str) -> tuple[float, float, dict]:
    usable = []
    for row in rows:
        true_label = normalize_label(row.get(label_col, ""))
        if true_label:
            usable.append((row["sentiment_mean_weighted"], true_label))
    if not usable:
        return 0.33, -0.33, {}

    best = {"macro_f1": -1.0, "pos_thr": 0.33, "neg_thr": -0.33, "per_class": {}, "confusion_matrix": {}}
    scores = [s for s, _ in usable]
    y_true = [y for _, y in usable]

    for pos_thr in [x / 100 for x in range(5, 70, 2)]:
        for neg_thr in [x / 100 for x in range(-70, -4, 2)]:
            if neg_thr < pos_thr:
                y_pred = []
                for s in scores:
                    if s > pos_thr:
                        y_pred.append("POSITIVE")
                    elif s < neg_thr:
                        y_pred.append("NEGATIVE")
                    else:
                        y_pred.append("NEUTRAL")
                m_f1, per_cls = macro_f1(y_true, y_pred)
                if m_f1 > best["macro_f1"]:
                    best["macro_f1"] = m_f1
                    best["pos_thr"] = round(pos_thr, 4)
                    best["neg_thr"] = round(neg_thr, 4)
                    best["per_class"] = per_cls
                    best["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return best["pos_thr"], best["neg_thr"], best


def run_inference(df: pd.DataFrame, text_col: str, pipe, low_conf_thr: float, no_preprocess: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    review_data: list[dict] = []
    all_sentences: list[str] = []
    sentence_map: list[tuple[int, int]] = []
    parse_errors: list[dict] = []
    cols = set(df.columns)

    for _, row in df.iterrows():
        review_idx = len(review_data)
        raw_val = row[text_col]
        orig_segmented = (
            row["review_text_sentence_segmented"]
            if "review_text_sentence_segmented" in cols and text_col != "review_text_sentence_segmented"
            else raw_val
        )
        segments, parse_error = clean_and_parse_segments(raw_val, no_preprocess)
        if parse_error:
            parse_errors.append(
                {"id": row["id"], "raw_text": str(raw_val), "parse_error": parse_error}
            )
        review_data.append(
            {
                "id": row["id"],
                "review_text_sentence_segmented": orig_segmented,
                "segments": segments,
                "results": {"labels": [], "scores": [], "confs": [], "probs": []},
                "orig_row": row.to_dict(),
            }
        )
        for s_idx, sent in enumerate(segments):
            all_sentences.append(sent)
            sentence_map.append((review_idx, s_idx))

    low_conf_rows: list[dict] = []
    predictions = pipe(all_sentences) if all_sentences else []

    for pred, (r_idx, _) in zip(predictions, sentence_map):
        label_prob = {LABEL_NORMALIZE.get(p["label"], "NEU"): float(p["score"]) for p in pred}
        norm_probs = {
            "NEG": label_prob.get("NEG", 0.0),
            "NEU": label_prob.get("NEU", 0.0),
            "POS": label_prob.get("POS", 0.0),
        }
        norm_label = max(norm_probs, key=norm_probs.get)
        conf = norm_probs[norm_label]
        raw_score = SCORE_MAP[norm_label]
        adjusted = adjust_score_with_rules(review_data[r_idx]["segments"][len(review_data[r_idx]["results"]["scores"])], raw_score)

        adjusted_label = "POS" if adjusted > 0 else ("NEG" if adjusted < 0 else "NEU")
        review_data[r_idx]["results"]["labels"].append(adjusted_label)
        review_data[r_idx]["results"]["scores"].append(round(adjusted, 4))
        review_data[r_idx]["results"]["confs"].append(round(conf, 4))
        review_data[r_idx]["results"]["probs"].append(norm_probs)

        if conf < low_conf_thr:
            low_conf_rows.append(
                {
                    "id": review_data[r_idx]["id"],
                    "clause_text": review_data[r_idx]["segments"][len(review_data[r_idx]["results"]["scores"]) - 1],
                    "pred_label": norm_label,
                    "confidence": round(conf, 4),
                    "probs": json.dumps(norm_probs, ensure_ascii=False),
                }
            )
    return review_data, parse_errors, low_conf_rows


def finalize_rows(review_data: list[dict], pos_thr: float, neg_thr: float) -> list[dict]:
    output_rows = []
    for item in review_data:
        res = item["results"]
        feats = compute_review_features(res["scores"], res["confs"], res["probs"])
        mean_w = feats["sentiment_mean_weighted"]
        if mean_w > pos_thr:
            final_label = "POSITIVE"
        elif mean_w < neg_thr:
            final_label = "NEGATIVE"
        else:
            final_label = "NEUTRAL"
        merged_row = dict(item.get("orig_row", {}))
        merged_row.update({
            "id": item["id"],
            "review_text_sentence_segmented": item["review_text_sentence_segmented"],
            "review_text_cleaned": " || ".join(item["segments"]),
            "clause_labels": json.dumps(res["labels"], ensure_ascii=False),
            "clause_scores": json.dumps(res["scores"], ensure_ascii=False),
            "clause_probs": json.dumps(res["probs"], ensure_ascii=False),
            "num_sentences": feats["num_sentences"],
            "num_pos": feats["num_pos"],
            "num_neg": feats["num_neg"],
            "num_neu": feats["num_neu"],
            "sum_score": feats["sum_score"],
            "sentiment_mean_weighted": feats["sentiment_mean_weighted"],
            "avg_confidence": feats["avg_confidence"],
            "has_mixed": feats["has_mixed"],
            "prob_pos": feats["prob_pos"],
            "prob_neu": feats["prob_neu"],
            "prob_neg": feats["prob_neg"],
            "final_label": final_label,
        })
        output_rows.append(merged_row)
    return output_rows


def main():
    args = parse_args()
    start = time.time()
    print("=" * 60)
    print("VISObert sentiment pipeline (enhanced)")
    print(f"Device: {'GPU' if DEVICE >= 0 else 'CPU'}")
    print("=" * 60)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_file)
    parse_log_path = Path(args.parse_log_file)
    low_conf_path = Path(args.low_conf_file)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    required_cols = {"id", args.text_col}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}. Cột hiện có: {list(df.columns)}")

    if args.limit and args.limit > 0:
        print(f"Limiting execution to first {args.limit} rows for testing.")
        df = df.head(args.limit)

    print(f"Loading model: {MODEL_NAME}")
    pipe = build_pipeline(args.batch_size, args.max_length)
    print(f"Running inference for {len(df):,} reviews...")
    review_data, parse_errors, low_conf_rows = run_inference(
        df=df,
        text_col=args.text_col,
        pipe=pipe,
        low_conf_thr=args.low_conf_thr,
        no_preprocess=args.no_preprocess,
    )

    pos_thr, neg_thr = 0.33, -0.33
    report = {
        "model_name": MODEL_NAME,
        "default_thresholds": {"pos_thr": 0.33, "neg_thr": -0.33},
        "optimized_thresholds": None,
        "validation_metrics": None,
    }

    if args.val_file:
        val_df = pd.read_csv(args.val_file, dtype=str, keep_default_na=False)
        val_df.columns = val_df.columns.str.strip()
        val_required = {"id", args.text_col, args.val_label_col}
        val_missing = sorted(val_required - set(val_df.columns))
        if val_missing:
            raise ValueError(f"Validation file thiếu cột: {val_missing}")

        val_review_data, _, _ = run_inference(
            df=val_df,
            text_col=args.text_col,
            pipe=pipe,
            low_conf_thr=args.low_conf_thr,
            no_preprocess=args.no_preprocess,
        )
        val_rows_default = finalize_rows(val_review_data, 0.33, -0.33)
        rows_with_label = []
        for r, (_, src) in zip(val_rows_default, val_df.iterrows()):
            merged = dict(r)
            merged[args.val_label_col] = src[args.val_label_col]
            rows_with_label.append(merged)

        pos_thr, neg_thr, best = optimize_thresholds(rows_with_label, args.val_label_col)
        report["optimized_thresholds"] = {"pos_thr": pos_thr, "neg_thr": neg_thr}
        report["validation_metrics"] = {
            "macro_f1": best.get("macro_f1", 0.0),
            "per_class": best.get("per_class", {}),
            "confusion_matrix": best.get("confusion_matrix", {}),
        }
        print(f"Optimized thresholds from validation: pos>{pos_thr}, neg<{neg_thr}")
    else:
        print("No validation file provided. Using default thresholds: +0.33 / -0.33")

    final_rows = finalize_rows(review_data, pos_thr=pos_thr, neg_thr=neg_thr)
    final_df = pd.DataFrame(final_rows)

    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if parse_errors:
        pd.DataFrame(parse_errors).to_csv(parse_log_path, index=False, encoding="utf-8-sig")
    if low_conf_rows:
        pd.DataFrame(low_conf_rows).to_csv(low_conf_path, index=False, encoding="utf-8-sig")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    if parse_errors:
        print(f"Parse errors logged: {parse_log_path} ({len(parse_errors)})")
    if low_conf_rows:
        print(f"Low-confidence samples logged: {low_conf_path} ({len(low_conf_rows)})")


if __name__ == "__main__":
    main()