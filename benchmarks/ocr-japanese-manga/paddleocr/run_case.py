#!/usr/bin/env python3

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR against one benchmark case.")
    parser.add_argument(
        "--case",
        required=True,
        help="Case id under benchmarks/ocr-japanese-manga/cases/",
    )
    parser.add_argument(
        "--image",
        default="source.png",
        help="Benchmark image filename inside the case directory. Default: source.png",
    )
    parser.add_argument(
        "--lang",
        default="japan",
        help="PaddleOCR language code. Default: japan",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Paddle device string. Default: gpu:0",
    )
    parser.add_argument(
        "--det-model",
        default="PP-OCRv5_server_det",
        help="Text detection model name. Default: PP-OCRv5_server_det",
    )
    parser.add_argument(
        "--rec-model",
        default="PP-OCRv5_server_rec",
        help="Text recognition model name. Default: PP-OCRv5_server_rec",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory. Default: ./outputs",
    )
    parser.add_argument(
        "--disable-layout-postprocess",
        action="store_true",
        help="Disable Japanese vertical reading-order postprocessing.",
    )
    return parser.parse_args()


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

    joined_text = "\n".join(block["text"] for block in blocks)
    return blocks, joined_text


def sort_horizontal_blocks(blocks: list[dict]) -> list[dict]:
    return sorted(
        blocks,
        key=lambda block: (
            block.get("bbox", {}).get("y", math.inf),
            block.get("bbox", {}).get("x", math.inf),
        ),
    )


def vertically_related(a: dict, b: dict) -> bool:
    if not a.get("bbox") or not b.get("bbox"):
        return False
    ab = a["bbox"]
    bb = b["bbox"]
    ay1, ay2 = ab["y"], ab["y"] + ab["height"]
    by1, by2 = bb["y"], bb["y"] + bb["height"]
    overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    min_h = max(1.0, min(ab["height"], bb["height"]))
    x_gap = abs(ab["x"] - bb["x"])
    width_ref = max(ab["width"], bb["width"])
    return overlap / min_h >= 0.45 and x_gap <= width_ref * 1.6


def cluster_vertical_blocks(blocks: list[dict]) -> list[list[dict]]:
    verticals = [
        block for block in blocks
        if block.get("direction") == "vertical"
        and block.get("script") in {"japanese", "mixed"}
        and block.get("bbox")
    ]
    clusters: list[list[dict]] = []
    for block in verticals:
        placed = False
        for cluster in clusters:
            if any(vertically_related(block, existing) for existing in cluster):
                cluster.append(block)
                placed = True
                break
        if not placed:
            clusters.append([block])
    return clusters


def local_vertical_reorder(blocks: list[dict]) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    cleaned = [block for block in blocks if not is_noise_block(block)]
    noise = [block for block in blocks if is_noise_block(block)]

    clusters = cluster_vertical_blocks(cleaned)
    clustered_ids = {block["id"] for cluster in clusters for block in cluster}
    non_vertical = [block for block in cleaned if block["id"] not in clustered_ids]

    ordered_clusters = sorted(
        clusters,
        key=lambda cluster: sum(block["bbox"]["x"] for block in cluster) / len(cluster),
        reverse=True,
    )

    result: list[dict] = []
    for cluster in ordered_clusters:
        result.extend(sorted(cluster, key=lambda item: (-item["bbox"]["x"], item["bbox"]["y"])))
    result.extend(sort_horizontal_blocks(non_vertical))

    original_order = [block["id"] for block in cleaned]
    if [block["id"] for block in result] != original_order:
        warnings.append("vertical_order_corrected")
    for block in noise:
        warnings.append(f"noise_filtered:{block['text']}")
    return result, warnings


def reorder_vertical_japanese_blocks(blocks: list[dict]) -> tuple[list[dict], list[str]]:
    return local_vertical_reorder(blocks)


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


def build_raw_artifact(raw_result: list[dict]) -> list[dict]:
    artifact: list[dict] = []
    for item in raw_result:
        if not isinstance(item, dict):
            artifact.append(to_json_safe(item))
            continue
        artifact.append(
            {
                "input_path": item.get("input_path"),
                "page_index": item.get("page_index"),
                "model_settings": item.get("model_settings"),
                "dt_polys": item.get("dt_polys"),
                "dt_scores": item.get("dt_scores"),
                "rec_texts": item.get("rec_texts"),
                "rec_scores": item.get("rec_scores"),
                "rec_polys": item.get("rec_polys"),
                "rec_boxes": item.get("rec_boxes"),
                "textline_orientation_angles": item.get("textline_orientation_angles"),
                "return_word_box": item.get("return_word_box"),
            }
        )
    return to_json_safe(artifact)


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    benchmark_root = repo_root / "benchmarks" / "ocr-japanese-manga"
    case_dir = benchmark_root / "cases" / args.case
    image_path = case_dir / args.image

    if not case_dir.exists():
        print(f"Case directory not found: {case_dir}", file=sys.stderr)
        return 2
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        print("paddleocr is not installed in this environment.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3

    output_dir = Path(args.output_dir) if args.output_dir else (Path(__file__).resolve().parent / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.case}.json"
    human_output_path = output_dir / f"{args.case}.human.json"
    raw_output_path = output_dir / f"{args.case}.raw.json"

    started = time.perf_counter()
    ocr = PaddleOCR(
        lang=args.lang,
        text_detection_model_name=args.det_model,
        text_recognition_model_name=args.rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=args.device,
    )
    raw_result = ocr.predict(str(image_path))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    raw_safe = build_raw_artifact(raw_result)
    blocks, joined_text = normalize_result(raw_safe)
    warnings: list[str] = []
    if not args.disable_layout_postprocess:
        blocks, warnings = reorder_vertical_japanese_blocks(blocks)
        joined_text = "\n".join(block["text"] for block in blocks)

    payload = {
        "engine": "paddleocr",
        "case_id": args.case,
        "image_path": str(image_path),
        "config": {
            "lang": args.lang,
            "device": args.device,
            "text_detection_model_name": args.det_model,
            "text_recognition_model_name": args.rec_model,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        "elapsed_ms": elapsed_ms,
        "joined_text": joined_text,
        "lines": blocks,
        "warnings": warnings,
        "artifacts": {
            "human": str(human_output_path),
            "raw": str(raw_output_path),
        },
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    human_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_output_path.write_text(
        json.dumps(
            {
                "engine": "paddleocr",
                "case_id": args.case,
                "image_path": str(image_path),
                "elapsed_ms": elapsed_ms,
                "raw": raw_safe,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {human_output_path}")
    print(f"Wrote {raw_output_path}")
    print(f"elapsed_ms={elapsed_ms}")
    print("--- text ---")
    print(joined_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
