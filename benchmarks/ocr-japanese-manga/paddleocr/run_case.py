#!/usr/bin/env python3

import sys
import time
from lib.artifacts import build_raw_artifact, write_outputs
from lib.cli import parse_args
from lib.layout import reorder_vertical_japanese_blocks
from lib.normalize import normalize_result
from lib.paths import resolve_case_paths


def main() -> int:
    args = parse_args()

    resolved = resolve_case_paths(args.case, args.image, args.output_dir)
    case_dir = resolved["case_dir"]
    image_path = resolved["image_path"]

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

    resolved["output_dir"].mkdir(parents=True, exist_ok=True)

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
            "human": str(resolved["human_output_path"]),
            "raw": str(resolved["raw_output_path"]),
        },
    }

    write_outputs(
        resolved["output_path"],
        resolved["human_output_path"],
        resolved["raw_output_path"],
        payload,
        {
            "engine": "paddleocr",
            "case_id": args.case,
            "image_path": str(image_path),
            "elapsed_ms": elapsed_ms,
            "raw": raw_safe,
        },
    )

    print(f"Wrote {resolved['output_path']}")
    print(f"Wrote {resolved['human_output_path']}")
    print(f"Wrote {resolved['raw_output_path']}")
    print(f"elapsed_ms={elapsed_ms}")
    print("--- text ---")
    print(joined_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
