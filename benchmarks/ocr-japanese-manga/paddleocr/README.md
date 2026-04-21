# PaddleOCR Smoke Test

This directory is the minimal local runner for Japanese manga OCR benchmark cases.

## Ownership

- Benchmark cases live under `../cases/`
- This directory owns only PaddleOCR-specific execution and output shaping
- It does not own scoring and it does not own gallery-app integration

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
