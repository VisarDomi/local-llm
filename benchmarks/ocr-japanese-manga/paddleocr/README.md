# PaddleOCR Smoke Test

This directory is the minimal local runner for Japanese manga OCR benchmark cases.

## Ownership

- Benchmark cases live under `../cases/`
- This directory owns only PaddleOCR-specific execution and output shaping
- It does not own scoring and it does not own gallery-app integration

## Shared Runtime

This benchmark does not own its own permanent Python runtime.

- Canonical venv: `~/.local/share/ocr/paddleocr-venv`
- Compatibility symlink from this repo: `/home/visar/Documents/work/ai/local-llm/.venv-paddleocr`

The Gallery Reader server also points at the same shared venv. The intent is:

- one heavyweight Paddle/CUDA runtime on disk
- benchmark harness reuses it
- product server reuses it

If the runtime does not exist yet, create or move it there first.

## Expected Output

Runner writes one JSON file per invocation into `./outputs/`.

The JSON is normalized for later comparison across OCR engines:

- `engine`
- `case_id`
- `image_path`
- `elapsed_ms`
- `joined_text`
- `lines`
- `raw`

## Seed Case

- `../cases/tanetsuke-delivery-cook-milk-page-10/source.png`
