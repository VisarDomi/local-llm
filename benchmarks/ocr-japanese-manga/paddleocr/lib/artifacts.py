import json
from pathlib import Path

from .normalize import to_json_safe


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


def write_outputs(
    output_path: Path,
    human_output_path: Path,
    raw_output_path: Path,
    payload: dict,
    raw_payload: dict,
) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    human_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_output_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
