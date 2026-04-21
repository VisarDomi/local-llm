import math

from .normalize import is_noise_block


def sort_horizontal_blocks(blocks: list[dict]) -> list[dict]:
    return sorted(
        blocks,
        key=lambda block: (
            (block.get("bbox") or {}).get("y", math.inf),
            (block.get("bbox") or {}).get("x", math.inf),
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


def reorder_vertical_japanese_blocks(blocks: list[dict]) -> tuple[list[dict], list[str]]:
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
