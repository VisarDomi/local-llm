# Decisions

## Hardware

- RTX 3060 12GB VRAM (360 GB/s bandwidth), Xeon E5-1650v3 6C/12T (Haswell 2014, no AVX-512), 32GB DDR4 2133MHz (~34 GB/s)
- CUDA 13.0 toolkit, driver 580.65.06, llama.cpp build b8736 (2026-04-09), sm_86
- No powercap — running at stock 170W. Max BIOS allows 212W.
- CPU is the bottleneck for parallel inference — token sampling happens on CPU per-slot sequentially. GPU at ~20% during parallel generation. Hard ceiling: ~350-370 t/s aggregate.
- Energy cost: ~270W under load (GPU 170W + system 100W). 24h/day × 30 days = 194 kWh/month = ~$19.40/month at $0.10/kWh.

## Architecture

- Three tiers, one runs at a time, switched via `llm switch <tier>`.
- No systemd services — the `llm` script manages the process directly (pidfile + health check).
- Tiers use different ports (8010/8100/8200) so harnesses don't need reconfiguration.
- CLI overrides on `llm switch` replace matching tier defaults.
- llama-server is the runtime. Serves OpenAI-compatible API.
- `--reasoning off` is mandatory for all tiers — without it, thinking tokens consume the entire generation budget on small models and produce empty responses.

## Benchmarked Models (quality from polecat suite, speed from real context tests)

| Model | Quant | Size | TG t/s @34K | PP t/s @34K | Quality | VRAM peak @200K |
|-------|-------|------|-------------|-------------|---------|-----------------|
| 0.8B | Q8_0 | 764MB | 148 | 6,913 | 5/25 | ~2.8GB |
| 2B | Q8_0 | 1.86GB | 96 | 4,972 | 9/25 | ~3.9GB |
| 4B | Q8_0 | 4.48GB | 43 | 2,060 | 10/25 | 11.1GB (Q8 KV, 262K) |
| **9B** | **IQ4_NL** | **5.37GB** | **41** | **1,518** | **15/25** | **11.5GB (Q8 KV, 262K)** |
| 27B | UD-IQ2_M | 9.49GB | 19 | ~300 | 14/25 | ~11.2GB |
| **35B-A3B** | **IQ4_NL** | **17.8GB** | **21 (regex)** | **420 (regex)** | **17/25** | **11.9GB (regex+Q8 KV, 262K)** |
| Haiku 4.5 | API | N/A | ~100 | N/A | **22/25** | N/A ($0.80/M) |

## Context Scaling (real measured data, clean restart per test)

### 4B Q8_0 + Q8 KV + 262K context
| Context | PP t/s | TG t/s | VRAM |
|---------|--------|--------|------|
| 34K | 2,060 | 43 | 10,468MB |
| 93K | 1,506 | 32 | 10,712MB |
| 200K | 1,033 | 22 | 11,141MB |

### 9B IQ4_NL + Q8 KV + 262K context
| Context | PP t/s | TG t/s | VRAM |
|---------|--------|--------|------|
| 34K | 1,518 | 41 | 10,807MB |
| 93K | 1,178 | 30 | 11,073MB |
| 200K | 872 | 21 | 11,454MB |

### 35B-A3B IQ4_NL + regex (layers 0-35 CPU, 36-39 GPU) + Q8 KV + 262K
| Context | PP t/s | TG t/s | VRAM |
|---------|--------|--------|------|
| 34K | 420 | 21 | 11,589MB |
| 93K | 382 | 16 | 11,750MB |
| 200K | ~254 | ~14 | 11,896MB |

All models fit 200K+ context in 12GB VRAM with Q8 KV cache. No spilling.
PP and TG degrade ~2x from 34K to 200K across all models.

## Quality Findings

- 0.8B-4B are not viable for agentic coding. Fabricate bugs, wrong frameworks.
- 9B is the minimum for code generation. Perfect Svelte 5 runes (5/5) but can't analyze bugs.
- 4B Q8 and 9B IQ4_NL have nearly identical TG speed (~42 t/s) — 9B wins on quality, 4B has no reason to exist.
- 27B dense at IQ2_M scores LOWER than 35B MoE at IQ2_M (14 vs 17). MoE preserves knowledge better under aggressive quant.
- 35B-A3B IQ4_NL is best local model (17/25). Found response.ok bug, correct Svelte 5, solid retry code.
- Haiku 4.5 is the quality baseline (22/25). Only model that found the 30s timing race.
- Quant matters more than model size at small scales: 0.8B Q8 > 2B IQ2.

## Tiers (final, data-driven)

### Interactive (41 t/s, port 8100)
- Qwen3.5-9B IQ4_NL (5.37GB) + Q8 KV + 262K context
- Fits fully in VRAM with 200K context (11.5GB peak)
- Best for: code generation, Svelte 5, tool calling, quick tasks
- Weakness: can't analyze bugs or diagnose complex issues (15/25)
- Command: `llama-server -m qwen3.5-9b-iq4nl.gguf -ngl 99 --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 262144`

### Workhorse (port 8010) — dual mode
- Qwen3.5-35B-A3B IQ4_NL (17.82GB)
- **Fast mode (regex)**: PP 430 / TG 21 t/s. Layers 0-35 experts CPU, 36-39 GPU. 11.5GB VRAM. OOMs above ~188K context.
- **Full context mode (-cmoe)**: PP 309 / TG 15 t/s. All experts CPU. 6.7GB VRAM. Handles 262K.
- Strategy: regex for single-project loads (<150K), -cmoe for multi-project loads (>150K)
- Best for: code review, bug finding, deeper analysis (17/25)
- Fast command: `llama-server -m qwen3.5-35b-a3b-iq4nl.gguf -ngl 99 -ot "blk\.([0-2][0-9]|3[0-5])\.ffn_.*_exps\.weight=CPU" --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 188000`
- Full command: `llama-server -m qwen3.5-35b-a3b-iq4nl.gguf -ngl 99 -cmoe --no-mmap --jinja --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 -c 262144`

### Speed tier — RETIRED
- 0.8B-4B quality too low for agentic work
- Use API fallback instead: Haiku ($0.80/M, 22/25 quality) or Groq (500+ t/s)

### Planner — DEFERRED
- Qwen3.5-122B-A10B UD-IQ3_XXS (~44.7GB) — needs NVMe swap with 32GB RAM
- Leave for overnight batch jobs
- Upgrade path: add 32GB RAM ($35) to fit in memory

## Parallel Scaling (0.8B Q4_K_M, real concurrent requests)

| N | Aggregate t/s | Per-slot t/s | Wall time |
|---|--------------|-------------|-----------|
| 1 | 160 | 160 | 798ms |
| 2 | 252 | 126 | 1017ms |
| 3 | 281 | 94 | 1367ms |
| 4 | 310 | 78 | 1653ms |
| **5** | **343** | **69** | **1865ms** |
| 6 | 357 | 60 | 2154ms |
| 7 | 351 | 50 | 2550ms |
| 8 | 369 | 46 | 2775ms |

## Cache Hit (game changer)

Prompt cache is enabled by default in llama-server. Same-prefix follow-up queries skip prompt eval:
- Q1 (cold, 34K context): 5,386ms wall
- Q2 (cached, same prefix): 346ms wall (15x faster)
- Q3 (cached): 578ms wall

This means: load a project once (slow), then rapid-fire questions at ~350ms each.
For 35B on video-platform (140K tokens): first query ~15 min, follow-ups ~1-2s each.

## --no-mmap is Default

All tiers use --no-mmap. mmap causes slow/stalled loading on this hardware. Only exception: 122B planner (if it ever runs) may need mmap for NVMe-backed loading.

## Quant Quality Matters More Than Model Size

At small scales (0.8B-2B), quant quality is the dominant factor:
- 0.8B Q8 (decent quality) beats 2B IQ2 (garbage) on real code analysis
- 0.8B IQ2 and 2B IQ2 both produce repeating garbage on 34K context
- Always use the highest quant that fits in VRAM

## Flags Tested (none improve speed)

- Flash attention: no measurable difference (auto-enabled)
- KV cache quant (q8_0, q4_0): slightly HURTS speed (~5%)
- --direct-io: no effect on inference speed
- --clear-idle: no effect on single requests
- --spec-self 1: segfaults on Qwen3.5 (incompatible with Gated DeltaNet)
- --spec-type ngram-mod: slightly slower (~2%)
- --reasoning-budget 0: bug #21487, ignored. Use --reasoning off instead.

## --reasoning off is Critical

- Without it: thinking tokens consume entire generation budget, responses are empty
- With it: models produce actual output, quality dramatically improves
- Per-request override: `"chat_template_kwargs": {"enable_thinking": false}` in API body
- Pi harness approach: sends enable_thinking per-request based on task type

## Engine

- llama.cpp is the only engine. GGUF is the only format.
- ExLlamaV2 archived. ExLlamaV3 slower on Ampere (confirmed by turboderp, issue #144).
- TensorRT-LLM: FP8 unavailable on RTX 3060 (cc 8.6 needs 8.9+). Not worth it.
- vLLM/SGLang: not practical for 12GB single-user.

## Project Sizes (real tokenizer counts from Qwen3.5 tokenizer)

| Project | Tokens | Fits in 262K? |
|---------|--------|---------------|
| km-explorer | 33,886 | Easy |
| cashback-v3 | 35,171 | Easy |
| gallery-reader | 46,685 | Easy |
| manga-reader | 56,192 | Easy |
| video-platform | 97,925 | Yes |
| trader | 154,047 | Yes |
| **Total** | **424K** | No (need subsets) |

## Context Scale Test Files (real tokenizer counts)

| File | Projects | Real Tokens |
|------|----------|-------------|
| context-scale-35k.json | km-explorer | 35,345 |
| context-scale-120k.json | km + cashback-v3 + gallery | 119,755 |
| context-scale-177k.json | km + cashback-v3 + gallery + manga | 177,454 |
| context-scale-211k.json | video-platform + manga + gallery | 210,666 |
| context-scale-241k.json | trader + video-platform | 240,539 |

All unique project combos. Token counts verified by model tokenizer, not char/4 estimate.

## Test Suite Design (next session)

Per-project test suites with same categories:
1. **Context dump + analysis** — dump one project's source, ask analytical questions
2. **Git diff replay** — give "before" state + task description, model produces a fix. Opus evaluates correctness (not diff similarity).
3. **Cross-project** — apply a fix from one project to another (e.g., SW fetch handler fix: cashback → km-explorer)
4. **Tool calling** — give tools (grep, read_file), ask model to navigate and find issues

Interesting git patterns found across projects:
- Ownership refactors (video-platform, gallery-reader, km-explorer)
- Svelte 5 reactivity bugs (gallery-reader, trader)
- SW fetch handler bug (same fix in cashback AND km-explorer)
- Performance bugs (manga-reader CloakBrowser GPU spin)

Prompt processing speed (measured):
- 140K tokens at 35B IQ4_NL (regex, ~380 PP t/s): ~6 min for first token
- 140K tokens at 9B IQ4_NL (~1200 PP t/s): ~2 min for first token
- 60K tokens at 9B IQ4_NL: ~40s for first token
- After first load: cache hit, follow-ups ~350ms-2s depending on model
- 60K tokens at 35B = ~6 min, at 9B = ~1.2 min

## API Fallback

- DeepSeek V3.2 direct: $0.28/M input (cached: $0.028/M) — cheapest frontier reasoning
- MiMo-V2-Flash on OpenRouter: $0.09/M input — cheapest overall
- Groq: 500-1000 t/s server-side, Qwen3-32B at 662 t/s
- Haiku 4.5: $0.80/M input — minimum viable agentic quality (22/25)

## PRs to Watch

- #20700: Qwen3.5 native MTP — built-in speculative decoding, could 1.5-2x generation speed
- #21594: --reasoning-budget fix
- #18039: EAGLE-3 speculative decoding
- #21038: Activation rotation (already merged, improves quant quality)

## Max Context Test (241K tokens, trader + video-platform)

| Model | PP t/s | TG t/s | Wall | OOM? |
|-------|--------|--------|------|------|
| 9B IQ4_NL (Q8 KV) | 778 | 19.2 | 363s | No |
| 9B Heretic i1-IQ4_NL (Q8 KV) | 787 | 18.4 | 361s | No |
| 35B-A3B IQ4_NL regex (Q8 KV) | — | — | — | OOM at 188K |
| 35B-A3B IQ4_NL -cmoe (Q8 KV) | 325 | 12.1 | 659s | No |

## Heretic vs Base 9B Quality (Opus evaluation, 14/20 tie)

- Heretic: better presentation (code quotes, markdown structure, line numbers). But misdiagnoses WriteGate — fabricates a race that cancel-on-acquire prevents.
- Base: plainer but correctly identifies real timing window (user interaction during restore). More trustworthy analysis.
- Both get line numbers wrong by 80-100+ lines. Neither understands ownership patterns.
- Verdict: base 9B edges ahead for code review — correctness > presentation.
- Full eval saved: benchmarks/results/20260410-heretic-vs-base-9b-quality.json

## Testing Flow

- All tests use `run-test.sh <file.json>` wrapper (in benchmarks/)
- Thinking OFF by default (injects `chat_template_kwargs.enable_thinking=false`)
- Pass `--thinking` flag to test with thinking ON
- Server must be started with `--reasoning off` globally
- Without this: thinking tokens consume entire budget, responses are empty

## Test Suite Structure

```
benchmarks/
├── run-test.sh                           # wrapper: reasoning-off, saves results permanently
├── results/                              # permanent: every test response + perf saved as JSON
├── stats/                                # session summaries with all measured numbers
├── polecat-test-prompts.sh               # quick 5-test speed/quality check
├── context-scale-{35k,120k,177k,211k,241k}.json  # context scaling tests (real token counts)
├── video-platform/                       # 98K tokens, 651 commits
│   ├── context-dump.json
│   ├── replay-ownership.json             # AliasRegistry rewrite
│   ├── replay-gesture-fix.json           # deadzone ratio fix
│   ├── replay-alias-registry.json        # AliasManager → AliasRegistry
│   └── tool-call.json
├── manga-reader/                         # 60K tokens, 85 commits
│   ├── context-dump.json
│   ├── replay-page-pool.json
│   ├── replay-gpu-spin.json              # CloakBrowser 545%→7%
│   ├── replay-pagination.json
│   └── tool-call.json
├── gallery-reader/                       # 54K tokens, 80 commits
│   ├── context-dump.json
│   ├── replay-effect-retrigger.json      # Svelte 5 $effect bug
│   ├── replay-deduplicate.json           # DRY refactor
│   ├── replay-sprite-ownership.json
│   └── tool-call.json
├── km-explorer/                          # 39K tokens, 25 commits
│   ├── context-dump.json
│   ├── replay-scroll-restore.json        # iOS app-switch bug
│   ├── replay-navstack.json              # ownership-based nav
│   ├── replay-sw-fetch.json              # SW fetch handler bug
│   └── tool-call.json
└── cross-project/
    └── sw-fetch-fix.json                 # apply cashback fix to km-explorer
```

Eval approach: git diffs are NOT ground truth for exact match. They generate realistic task descriptions. Opus evaluates "did the model solve the same problem?" not "did it match the diff?"

## Next Steps

1. ~~Build per-project test suites from git history~~ DONE
2. ~~Run context scaling on 9B, 35B~~ DONE
3. ~~Test regex vs -cmoe on 35B~~ DONE (regex 39% faster, OOMs >188K; -cmoe slower, handles 262K)
4. ~~Test heretic vs base 9B~~ DONE (tie 14/20, base more accurate)
5. Run more heretic quality tests (different tasks, not just code review)
6. Run full per-project test suite on base 9B and 35B (git replay, tool calling, cross-project)
7. Set up Pi harness connecting to llm switcher
8. Test PR #20700 (MTP) — fork, cherry-pick, rebuild, benchmark
9. Consider testing Qwen3.5-35B-A3B Q8_0 (36.9GB) — fits in 32GB RAM with -cmoe

## Directory Structure

- ~/Documents/llm/ — self-contained
- models/ — GGUF files (12 models downloaded, ~80GB total)
- engine/ — llama.cpp git clone + build (b8736, CUDA 13.0)
- benchmarks/ — test files, run-test.sh, results/, stats/
- harness/ — Pi, Gastown, OpenCode (not yet set up)
- decisions.md — this file, single source of truth
