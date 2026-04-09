# Session Results — 2026-04-09

## Hardware
RTX 3060 12GB, Xeon E5-1650v3 6C/12T, 32GB DDR4 2133MHz
llama.cpp build b8736, CUDA 13.0, sm_86

## Single-Stream Benchmarks (llama-bench, synthetic pp512+tg128)

| Model | Quant | Size | PP t/s | TG t/s |
|-------|-------|------|--------|--------|
| 0.8B | Q4_K_M | 497MB | 8,703 | 238 |
| 0.8B | Q8_0 | 764MB | 8,741 | 207 |
| 2B | Q4_K_M | 1.18GB | 5,830 | 160 |
| 2B | Q8_0 | 1.86GB | 5,833 | 117 |
| 4B | Q4_K_M | 2.54GB | 2,543 | 79 |
| 4B | Q2_K_XL | 1.80GB | 2,253 | 79 |
| 9B | Q5_K_M | 6.12GB | 1,649 | 45 |

## Real Context Benchmarks (llama-server, km-explorer 34K tokens)

| Model | Quant | PP t/s | TG t/s | Wall | Quality |
|-------|-------|--------|--------|------|---------|
| 0.8B | IQ2_XXS | 6,873 | 164 | 11s | Garbage — repeating loop |
| 0.8B | Q8_0 | 6,913 | 148 | 12s | Decent — finds race conditions |
| 2B | IQ2_XXS | 4,889 | 123 | 15s | Garbage — "index handles index" loop |
| 2B | Q8_0 | 4,972 | 96 | 18s | Good — cites line numbers, AbortSignal |

Key finding: quant matters more than model size at small scales. 0.8B Q8 beats 2B IQ2.

## Cache Hit Test (0.8B Q8, km-explorer 34K context)

| Query | Wall time | Prompt processed | PP time |
|-------|-----------|-----------------|---------|
| Q1 (cold) | 5,386ms | 33,880 tok (full) | 4,910ms |
| Q2 (cached) | 346ms | 519 tok (diff only) | 121ms |
| Q3 (cached) | 578ms | 517 tok (diff only) | 121ms |

15x faster on cache hit. Load project once, then ~350ms per follow-up question.

## Real Context Benchmarks (llama-server, video-platform 52K tokens)

| Model | Quant | PP t/s | TG t/s | Wall |
|-------|-------|--------|--------|------|
| 0.8B | Q4_K_M | 5,969 | 142 | 12s |
| 0.8B | IQ2_XXS | 6,030 | 147 | 12s |
| 0.8B | Q8_0 | 6,094 | 132 | 13s |

Quant barely affects speed at 0.8B — model too small to be bandwidth-limited.

## Polecat Quality Tests (5 agentic tasks, scored by Opus 1-5 each)

| Model | Quant | T1 grep | T2 bugs | T3 svelte5 | T4 diag | T5 retry | Total |
|-------|-------|---------|---------|------------|---------|----------|-------|
| 0.8B | Q8_0 | 1 | 1 | 1 | 1 | 1 | 5/25 |
| 2B | Q8_0 | 1 | 1 | 1 | 1 | 2 | 6/25 |
| 4B | Q4_K_M | 2 | 1 | 1 | 2 | 4 | 10/25 |
| 9B | Q5_K_M | 3 | 2 | 5 | 2 | 3 | 15/25 |
| 27B | IQ2_M | 3 | 3 | 2 | 2 | 4 | 14/25 |
| 35B-A3B | IQ2_M | 3 | 4 | 4 | 2 | 4 | 17/25 |
| 35B-A3B | IQ3_XXS | 3 | 4 | 4 | 2 | 4 | ~18/25 |
| Haiku 4.5 | API | 4 | 5 | 3 | 5 | 5 | 22/25 |

## Parallel Scaling (0.8B Q4_K_M, real concurrent requests, 128 tokens)

| N | Aggregate t/s | Per-slot t/s | Wall |
|---|--------------|-------------|------|
| 1 | 160 | 160 | 798ms |
| 2 | 252 | 126 | 1017ms |
| 3 | 281 | 94 | 1367ms |
| 4 | 310 | 78 | 1653ms |
| 5 | 343 | 69 | 1865ms |
| 6 | 357 | 60 | 2154ms |
| 7 | 351 | 50 | 2550ms |
| 8 | 369 | 46 | 2775ms |

N=5 sweet spot. CPU sampling is the ceiling at ~350-370 t/s aggregate.

## 35B-A3B MoE Offload Tests

| Config | VRAM | TG t/s |
|--------|------|--------|
| -cmoe (all experts CPU) | 2.9GB | 21 |
| -ncmoe 30 | 5.8GB | 24 |
| -ncmoe 25 | OOM | - |
| -ncmoe 20 | OOM | - |

## Flags Tested (llama-server, single-stream)

| Flag | Effect on TG t/s |
|------|-----------------|
| Flash attention | No change (auto-enabled) |
| --cache-type-k q8_0 | -5% (hurts) |
| --cache-type-k q4_0 | -5% (hurts) |
| --direct-io | No change |
| --clear-idle | No change (single request) |
| --spec-self 1 | Segfault (Gated DeltaNet incompatible) |
| --spec-type ngram-mod | -2% (slight overhead) |
| --reasoning-budget 0 | Ignored (bug #21487) |
| --reasoning off | Critical — fixes empty responses |

## Models Downloaded

| File | Size | Location |
|------|------|----------|
| qwen3.5-0.8b-q4km.gguf | 508MB | ~/Documents/llm/models/ |
| qwen3.5-0.8b-q8.gguf | 775MB | ~/Documents/llm/models/ |
| qwen3.5-0.8b-iq2xxs.gguf | 323MB | ~/Documents/llm/models/ |
| qwen3.5-2b-q4km.gguf | 1.2GB | ~/Documents/llm/models/ |
| qwen3.5-2b-q8.gguf | 1.9GB | ~/Documents/llm/models/ |
| qwen3.5-2b-iq2xxs.gguf | 680MB | ~/Documents/llm/models/ |
| qwen3.5-4b-q4km.gguf | 2.6GB | ~/Documents/llm/models/ |
| qwen3.5-4b-q2kxl.gguf | 1.9GB | ~/Documents/llm/models/ |
| qwen3.5-9b-q5km.gguf | 6.2GB | ~/Documents/llm/models/ |
| qwen3.5-27b-iq2m.gguf | 9.5GB | ~/Documents/llm/models/ |
| qwen3.5-35b-a3b-iq2m.gguf | 10.6GB | ~/Documents/llm/models/ |
| qwen3.5-35b-a3b-iq3xxs.gguf | 12.2GB | ~/Documents/llm/models/ |

## 4B Q8_0 Context Scaling (clean restart each, Q8 KV cache, 262K max context)

| Context | Tokens | PP t/s | TG t/s | Wall | VRAM idle→peak |
|---------|--------|--------|--------|------|----------------|
| 34K | 33,944 | 2,060 | 43.3 | 40s | 10,334→10,468 (+134MB) |
| 48K | 52,782 | 1,871 | 39.6 | 41s | 10,334→10,542 (+208MB) |
| 93K | 97,983 | 1,506 | 32.0 | 97s | 10,334→10,712 (+378MB) |
| 100K | 100,740 | 1,484 | 31.4 | 85s | 10,330→10,724 (+394MB) |
| 200K | 200,741 | 1,033 | 22.0 | 218s | 10,330→11,141 (+811MB) |

VRAM at startup (model + empty KV): 10,334MB. KV grows ~4 bytes/token (Q8 quantized).
Full 200K fits in 12GB with 0.9GB to spare. No spilling.
PP and TG both degrade ~2x from 34K to 200K (linear with context length).

## 9B IQ4_NL Context Scaling (Q8 KV cache, 262K max context, clean restart each)

| Context | Tokens | PP t/s | TG t/s | Wall | VRAM peak |
|---------|--------|--------|--------|------|-----------|
| 34K | 33,944 | 1,518 | 40.6 | 48s | 10,807MB |
| 48K | 52,782 | 1,391 | 36.4 | 52s | 10,889MB |
| 93K | 97,983 | 1,178 | 30.1 | 118s | 11,073MB |
| 100K | 100,740 | 1,173 | 29.4 | 104s | 11,074MB |
| 200K | 200,741 | 872 | 21.0 | 255s | 11,454MB |

Model: 5.37GB. Q8 KV cache grows dynamically. Peak 11.5GB at 200K — fits with 0.5GB spare.
Quality excellent at all context lengths. 200K test identified 12 shared patterns across 3 projects.

## 35B-A3B IQ4_NL Context Scaling (regex offload layers 0-35→CPU, 36-39→GPU, Q8 KV, 262K)

Command: llama-server -m qwen3.5-35b-a3b-iq4nl.gguf -ngl 99 -ot "blk\.([0-2][0-9]|3[0-5])\.ffn_.*_exps\.weight=CPU" --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 262144

| Context | Tokens | PP t/s | TG t/s | Wall | VRAM peak |
|---------|--------|--------|--------|------|-----------|
| 34K | 33,944 | 420 | 21.3 | 129s | 11,589MB |
| 48K | 52,782 | 414 | 19.0 | 155s | 11,626MB |
| 93K | 97,983 | 382 | 15.6 | 322s | 11,750MB |
| 100K | 100,740 | 374 | 14.7 | 305s | 11,734MB |
| 200K | 200,741 | ~254 | ~14 | 788s | 11,896MB |

Model: 17.82GB (7.8GB on CPU, ~9.2GB on GPU). 4 layers' experts on GPU.
KV cache: Q8, 2720MB pre-allocated for 262K context.
200K test: 13 min wall time. VRAM peaked at 11.9GB (0.1GB spare).
PP degrades ~1.7x from 34K to 200K. TG degrades ~1.5x.

## 35B-A3B IQ4_NL Context Scaling (regex offload 4 GPU layers, Q8 KV, 262K context)

Command:
```
llama-server -m qwen3.5-35b-a3b-iq4nl.gguf -ngl 99 \
  -ot "blk\.([0-2][0-9]|3[0-5])\.ffn_.*_exps\.weight=CPU" \
  --no-mmap --jinja --reasoning off \
  --cache-type-k q8_0 --cache-type-v q8_0 -c 262144 --port 8200
```

Tests run with: `run-test.sh ~/Documents/llm/benchmarks/<file>.json`
Tests 1-3 clean (no cache). Tests 4-5: test 4 redone with clean restart. Test 5 estimated from 788s wall time.

| Context | Tokens | PP t/s | TG t/s | Wall | VRAM peak |
|---------|--------|--------|--------|------|-----------|
| 34K | 33,944 | 420 | 21.3 | 129s | 11,589MB |
| 48K | 52,782 | 414 | 19.0 | 155s | 11,626MB |
| 93K | 97,983 | 382 | 15.6 | 322s | 11,750MB |
| 100K | 100,740 | 374 | 14.7 | 305s | 11,734MB |
| 200K | 200,741 | ~270 | ~10 | 788s | 11,896MB |

Model: 17.82GB file, layers 0-35 experts on CPU, layers 36-39 on GPU.
VRAM: ~9.2GB model (GPU portion) + ~2.7GB KV (Q8, grows with context).
Peak 11.9GB at 200K — fits with 0.1GB spare. Very tight.

## 9B IQ3_XXS with -cmoe (from earlier today, partial data)

Command:
```
llama-server -m qwen3.5-9b-iq3xxs.gguf -ngl 99 --no-mmap --jinja --reasoning off \
  --cache-type-k q4_0 --cache-type-v q4_0 -c 262144 --port 8200
```

Test: `run-test.sh ~/Documents/llm/benchmarks/km-explorer/context-dump.json`

| Context | Tokens | PP t/s | TG t/s | Wall | VRAM peak |
|---------|--------|--------|--------|------|-----------|
| 34K | 33,944 | 1,469 | 41.9 | 48s | 7,511MB |

## 35B-A3B IQ4_NL 200K REAL DATA (-cmoe, not regex)

Command: llama-server -m qwen3.5-35b-a3b-iq4nl.gguf -ngl 99 -ot ".*_exps\.weight=CPU" --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 262144

| Context | Tokens | PP t/s | TG t/s | Wall |
|---------|--------|--------|--------|------|
| 200K | 200,741 | 325 | 12.07 | 659s |

## 35B-A3B IQ4_NL: regex vs -cmoe comparison (35K km-explorer, Q8 KV, 262K max context)

| Mode | PP t/s | TG t/s | VRAM at startup | Max usable context |
|------|--------|--------|-----------------|-------------------|
| regex (layers 0-35 CPU, 36-39 GPU) | 430 | 21.2 | 11,529MB | ~188K (OOMs above) |
| -cmoe (all experts CPU) | 309 | 14.6 | ~6,700MB | 262K+ |

Regex is 39% faster PP, 45% faster TG. But OOMs at ~188K context.
Strategy: regex for short context (<150K), -cmoe for full context (>150K).

## Context Scale Test Files (real tokenizer counts)

| File | Projects | Real Tokens |
|------|----------|-------------|
| context-scale-35k.json | km-explorer | 35,345 |
| context-scale-120k.json | km + cashback-v3 + gallery | 119,755 |
| context-scale-177k.json | km + cashback-v3 + gallery + manga | 177,454 |
| context-scale-211k.json | video-platform + manga + gallery | 210,666 |
| context-scale-241k.json | trader + video-platform | 240,539 |

## 9B IQ4_NL 241K context (new context-scale files, Q8 KV, 262K max)

Command: llama-server -m qwen3.5-9b-iq4nl.gguf -ngl 99 --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 262144

| Context | Tokens | PP t/s | TG t/s | Wall |
|---------|--------|--------|--------|------|
| 241K | 240,551 | 778 | 19.15 | 363s |

No OOM. Fits in 12GB VRAM.

## 9B HERETIC i1-IQ4_NL 241K context (Q8 KV, 262K max)

Command: llama-server -m qwen3.5-9b-heretic-i1-iq4nl.gguf -ngl 99 --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 262144

| Context | Tokens | PP t/s | TG t/s | Wall |
|---------|--------|--------|--------|------|
| 241K | 240,551 | 787 | 18.43 | 361s |

Nearly identical speed to base 9B IQ4_NL (778 PP, 19.15 TG).
