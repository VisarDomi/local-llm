# OCR Japanese Manga Benchmark

Minimal benchmark set for comparing OCR engines on Japanese manga pages and viewport crops.

## Layout

- `cases/<case-id>/source.webp` — original source image for the benchmark case
- `cases/<case-id>/source.png` — normalized image input for OCR runners
- `cases/<case-id>/case.json` — metadata and intended benchmark usage
- `cases/<case-id>/notes.md` — optional human notes about expected difficulty or failure modes

## Current Seed Case

- `tanetsuke-delivery-cook-milk-page-10`

This seed case is copied from local gallery storage so the benchmark corpus is self-contained.
