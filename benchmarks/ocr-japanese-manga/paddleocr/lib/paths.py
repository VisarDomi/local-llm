from pathlib import Path


def resolve_case_paths(case_id: str, image_name: str, output_dir_override: str | None) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    benchmark_root = repo_root / "benchmarks" / "ocr-japanese-manga"
    case_dir = benchmark_root / "cases" / case_id
    image_path = case_dir / image_name
    output_dir = Path(output_dir_override) if output_dir_override else (Path(__file__).resolve().parents[1] / "outputs")
    return {
        "repo_root": repo_root,
        "benchmark_root": benchmark_root,
        "case_dir": case_dir,
        "image_path": image_path,
        "output_dir": output_dir,
        "output_path": output_dir / f"{case_id}.json",
        "human_output_path": output_dir / f"{case_id}.human.json",
        "raw_output_path": output_dir / f"{case_id}.raw.json",
    }
