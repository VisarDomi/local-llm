import re
from pathlib import Path

JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
NOISE_RE = re.compile(r"^[A-Za-z0-9]$")


def polygon_to_bbox(box: object) -> dict[str, float] | None:
    if not isinstance(box, list) or not box:
        return None
    pts: list[tuple[float, float]] = []
    for pt in box:
        if not isinstance(pt, list) or len(pt) < 2:
            continue
        try:
            pts.append((float(pt[0]), float(pt[1])))
        except (TypeError, ValueError):
            continue
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {
        "x": min(xs),
        "y": min(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def infer_direction(text: str, bbox: dict[str, float] | None) -> str:
    if not bbox:
        return "unknown"
    width = max(1.0, bbox["width"])
    height = max(1.0, bbox["height"])
    ratio = height / width
    if ratio >= 2.2 and JP_RE.search(text):
        return "vertical"
    if width >= height:
        return "horizontal"
    return "unknown"


def infer_script(text: str) -> str:
    has_jp = bool(JP_RE.search(text))
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)
    if has_jp and has_latin:
        return "mixed"
    if has_jp:
        return "japanese"
    if has_latin:
        return "latin"
    return "unknown"


def is_noise_block(block: dict) -> bool:
    text = block.get("text", "").strip()
    bbox = block.get("bbox")
    if not text:
        return True
    if len(text) == 1 and NOISE_RE.fullmatch(text):
        return True
    if bbox and bbox["width"] <= 40 and bbox["height"] <= 40 and not JP_RE.search(text):
        return True
    return False


def normalize_result(raw_result: object) -> tuple[list[dict], str]:
    blocks: list[dict] = []

    if not isinstance(raw_result, list):
        return blocks, ""

    def add_block(text: object, score: object, box: object = None) -> None:
        text_str = str(text).strip()
        if not text_str:
            return
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None
        bbox = polygon_to_bbox(box)
        blocks.append(
            {
                "id": f"block-{len(blocks) + 1}",
                "text": text_str,
                "score": score_val,
                "box": box,
                "bbox": bbox,
                "direction": infer_direction(text_str, bbox),
                "script": infer_script(text_str),
            }
        )

    for item in raw_result:
        if not isinstance(item, dict):
            continue

        rec_texts = item.get("rec_texts")
        rec_scores = item.get("rec_scores")
        rec_polys = item.get("rec_polys")
        if isinstance(rec_texts, list):
            for idx, text in enumerate(rec_texts):
                score = rec_scores[idx] if isinstance(rec_scores, list) and idx < len(rec_scores) else None
                box = rec_polys[idx] if isinstance(rec_polys, list) and idx < len(rec_polys) else None
                add_block(text, score, box)
            continue

        dt_polys = item.get("dt_polys")
        if isinstance(rec_texts, list):
            for idx, text in enumerate(rec_texts):
                score = rec_scores[idx] if isinstance(rec_scores, list) and idx < len(rec_scores) else None
                box = dt_polys[idx] if isinstance(dt_polys, list) and idx < len(dt_polys) else None
                add_block(text, score, box)
            continue

        rec_text = item.get("rec_text")
        if rec_text is not None:
            add_block(rec_text, item.get("rec_score"), item.get("dt_polys"))
            continue

        if "ocr_res" in item and isinstance(item["ocr_res"], list):
            for entry in item["ocr_res"]:
                if not isinstance(entry, dict):
                    continue
                add_block(entry.get("text", ""), entry.get("score"), entry.get("poly"))

    return blocks, "\n".join(block["text"] for block in blocks)


def to_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return to_json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return repr(value)
