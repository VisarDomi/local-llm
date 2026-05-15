# Decisions

## 2026-05-15: Qwen3.6 35B Q6/Q8 on RTX 3060 12GB + 32GB RAM

Hardware for this pass:

- GPU: RTX 3060 12GB
- RAM: 32GB
- CPU: 6 physical cores
- Runtime: llama.cpp stable build with CUDA 13.1 env from `~/.config/cuda-env.sh`
- Goal: run Qwen3.6 35B MoE with no mmap/NVMe paging when possible.

### CUDA and Runtime

- CUDA 13.1.2 is the intended toolkit for the local llama.cpp path.
- The stable llama.cpp binary is built from the local stable engine under `engine/stable/llama.cpp`.
- Use `source ~/.config/cuda-env.sh` before manual tests.

### Placement Strategy

The useful strategy is explicit tensor placement, not code changes:

- Keep `-ngl 99` so non-overridden tensors are aggressively eligible for GPU.
- Use `-ot ...=CPU` to move selected MoE expert tensors back to CPU.
- Use `--no-mmap` when testing whether the model fits in RAM+VRAM without NVMe-backed mmap behavior.
- Use Q8 KV with `--cache-type-k q8_0 --cache-type-v q8_0`.

Current Q6 working split:

```bash
-ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
```

Meaning:

- GPU experts: layers `0,1,2,10,11,12,20,21`
- CPU experts: layers `3-9,13-19,22-39`

If higher context fails with CUDA OOM, the next lower-VRAM split is:

```bash
-ot "blk\.([2-9]|1[2-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
```

Meaning:

- GPU experts: layers `0,1,10,11,20`
- CPU experts: layers `2-9,12-19,21-39`

### Q8 Result

`Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf`:

- mmap + regex could load and answer at 262K.
- `--no-mmap` could not be made practical on 32GB RAM + 12GB VRAM.
- Around 8 GPU expert layers it was still RAM-bound.
- Around 10 GPU expert layers it became VRAM-bound and failed on an additional CUDA compute buffer.
- Conclusion: Q8 no-mmap is not viable on this machine.

Observed Q8 boundaries:

- 3 GPU expert layers: RAM-bound before load.
- 5 GPU expert layers: still RAM-bound, about 8.3GB VRAM used.
- 8 GPU expert layers: about 26.6GB cgroup memory, 27.9GB RSS, 9.9GB VRAM, still not loaded.
- 10 GPU expert layers: about 25.5GB cgroup memory, 26.6GB RSS, 11.6GB VRAM, then CUDA OOM on about 568MiB compute buffer.
- 12 GPU expert layers: immediate CUDA OOM, attempted about 12.6GiB CUDA allocation.

### Q6 Working Checkpoints

`Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` is the current viable candidate.

4K smoke test:

- `--no-mmap`
- `-ngl 99`
- current 8 GPU expert split
- `-c 4096`
- Q8 KV
- Answered `2+2 equals 4.`
- Prompt: 48.4 t/s
- Generation: 24.0 t/s
- Cgroup memory: about 25.9GB
- Process RSS: about 23.8GB
- VRAM: about 9.0GB used
- Global available RAM: about 5.9GB

40K server + 35K benchmark:

- Result file: `benchmarks/results/20260515-133923-qwen3.6-35b-q6-40k-nommap-35k.json`
- Prompt tokens: 35,345
- Completion tokens: 1,024
- Finish reason: `length`
- Wall time: 202,022ms
- Approx throughput from wall: 180 tokens/s across prompt plus completion
- Server loaded in about 60 seconds.
- Before benchmark: cgroup memory about 26.4GB, VRAM about 9.4GB, global available RAM about 5.8GB.
- During benchmark: resource use stayed stable around 9.4GB VRAM and 5.4GB available RAM.

40K server + 35K benchmark with default f16 KV:

- Command omitted explicit `--cache-type-k q8_0 --cache-type-v q8_0`, so llama.cpp used default f16 KV.
- Context: `-c 40000`, rounded to `n_ctx = 40192`.
- Prompt tokens: 35,345
- Completion tokens: 1,024
- Truncated: 0
- Prompt eval: 150,524.83ms, 234.81 tokens/s
- Generation eval: 46,296.95ms, 22.12 tokens/s
- Total time: 196,821.78ms for 36,369 tokens
- Interpretation: f16 KV preserved the same prompt-processing speed as Q8 KV at 35K and slightly improved generation speed, but it is expected to reduce the max context ceiling because it uses more VRAM.
- `--cache-ram 4096` did not load in this tight f16-KV shape, but `--cache-ram 512` loaded at 40K context.
- Interpretation: prompt cache is viable, but cache size must be tuned against RAM pressure. For 35K follow-ups, 512MiB should cover the observed checkpoint footprint because checkpoints were about 62.813MiB each and the 35K run created about 6 checkpoints.

Agent-coding f16 KV checkpoint:

- `CTX=131072`
- `CACHE_RAM=1024`
- `CHECKPOINT_EVERY=16384`
- `-np 1`
- `--no-mmap`
- default f16 KV
- current 8 GPU expert split
- This combination loads and gives useful cache hits at 120K prompt size.
- Resource usage during the run: about 11,776MiB / 12,288MiB VRAM and about 28.3GB RAM / 84.4%.
- Base 120K run:
  - Prompt tokens processed: 119,767
  - Completion tokens: 1,024
  - Truncated: 0
  - Prompt eval: 552,143.42ms, 216.91 tokens/s
  - Generation eval: 51,056.51ms, 20.06 tokens/s
  - Total time: 603,199.92ms for 120,791 tokens
- Checkpoint behavior:
  - Checkpoints every 16,384 tokens.
  - Checkpoint size remained about 62.813MiB.
  - The base run created checkpoints through the end of the 120K prompt.
- Same-prefix follow-up:
  - Slot selected by LCP similarity with `sim_best = 1.000`, `f_keep = 0.991`.
  - Restored checkpoint at 119,251 tokens.
  - Reprocessed only 548 prompt tokens.
  - Prompt eval: 3,683.49ms, 148.77 tokens/s
  - Generation eval: 25,461.90ms for 512 tokens, 20.11 tokens/s
  - Total follow-up time: 29,145.39ms for 1,060 tokens
- Interpretation: this is the current best agent-coding mode: f16 KV quality, 130K context, 1024MiB cache, one slot, no mmap, and fast follow-ups.
- Next pressure point is increasing context above 130K or preserving this cache behavior at higher context. Larger cache attempts can stall during `common_init_result: fitting params to device memory ...`.
- The 7 GPU expert regex loads with `CTX=163840`:
  - GPU experts: `0,1,10,11,12,20,21`
  - CPU experts: `2-9,13-19,22-39`
  - Regex: `blk\.([2-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU`
  - This is the next confirmed context step above the 8 GPU expert `CTX=131072` quality mode.

Full-context Qwen3.6 Q6 workhorse winner:

- `CTX=262144`
- `CACHE_RAM=2048`
- `CHECKPOINT_EVERY=16384`
- `CTX_CHECKPOINTS=64`
- `MemoryHigh=28672M`
- `MemoryMax=30720M`
- `MemorySwapMax=0`
- `-np 1`
- `--no-mmap`
- default f16 KV
- Live memory reading during the validated run:
  - System RAM: about `30GiB / 31GiB` used, `918MiB` available.
  - System swap: about `2.0GiB / 31GiB` used, but the llama scope had `MemorySwapCurrent=0` because `MemorySwapMax=0`.
  - Scope memory: `MemoryCurrent=29,897,916,416` bytes (`27.8GiB`), `MemoryPeak=30,040,580,096` bytes (`28.0GiB`), `MemoryAvailable=166,854,656` bytes (`159MiB`) before cgroup max.
  - Scope guard: `MemoryHigh=30,064,771,072` bytes (`28.0GiB`), `MemoryMax=32,212,254,720` bytes (`30.0GiB`).
  - VRAM: `11119MiB / 12288MiB` used, `790MiB` free, GPU utilization about `36%` after the run.
  - `vmstat` showed small swap-in (`96 KiB/s`) and no sampled swap-out, so the validated run was not actively thrashing at the sample point.
- 3 GPU expert split:
  - GPU experts: `0,10,20`
  - CPU experts: `1-9,11-19,21-39`
  - Regex: `blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU`
- 241K benchmark:
  - Prompt tokens processed: 240,551
  - Completion tokens: 1,024
  - Truncated: 0
  - Prompt eval: 1,339,566.37ms, 179.57 tokens/s
  - Generation eval: 64,808.75ms, 15.80 tokens/s
  - Total time: 1,404,375.12ms for 241,575 tokens
  - API result file: `benchmarks/results/20260516-001410-qwen3.6-q6-262144-241k.json`
  - Checkpoint progression reached `16/64` by the 241K prompt: regular checkpoints through coarse 16K spacing, then near-end tail checkpoints at 240,035 and 240,547 tokens.
- Follow-up cache probe:
  - Slot selected by LCP similarity with `sim_best = 1.000`, `f_keep = 0.996`.
  - Restored checkpoint at 240,035 tokens.
  - Reprocessed only 548 prompt tokens.
  - Prompt eval: 4,626.63ms, 118.44 tokens/s
  - Generation eval: 32,380.50ms for 512 tokens, 15.81 tokens/s
  - Total follow-up time: 37,007.13ms for 1,060 tokens
  - API reported `cached_tokens = 240,035`.
  - API result file: `benchmarks/results/20260516-001452-qwen3.6-q6-262144-241k-followup.json`
- Prompt-cache interpretation:
  - This is the current full-context workhorse. It preserves f16 KV quality, reaches model max context, keeps llama swap disabled, and has strong same-prefix follow-up caching.
  - `CACHE_RAM=2048` with `CHECKPOINT_EVERY=16384` and `CTX_CHECKPOINTS=64` produced 16 checkpoints for the 241K prompt. This halves regular checkpoint creation versus `2048/8192/32` while preserving the near-tail cache hit.
  - Do not model follow-up cost from regular spacing alone. llama.cpp also creates near-end checkpoints when checkpoint slots are available. In the 241K run, it restored from 240,035 tokens and reprocessed only 548 prompt tokens.
  - `CHECKPOINT_EVERY` is mainly the regular interval floor; near prompt end, llama.cpp deliberately creates tail checkpoints for better follow-up reuse.
  - Increasing `CTX_CHECKPOINTS` doubles the possible active-slot checkpoint storage only if `CACHE_RAM` is also raised. With the observed 62.813MiB checkpoint size, 64 checkpoints would need about 4020MiB for full coverage at 4096-token spacing.
  - Generation is slower than the 130K quality mode because only three expert layers remain on GPU, but cache behavior for sequential same-prefix follow-ups is good.

130K server + 120K benchmark:

- Prompt tokens processed: 119,767
- Completion tokens: 1,024
- Truncated: 0
- Prompt eval: 551,801.48ms, 217.05 tokens/s
- Generation eval: 61,130.88ms, 16.75 tokens/s
- Total time: 612,932.36ms for 120,791 tokens
- Live VRAM after load/test: about 10,775MiB / 12,288MiB
- Live RAM after load/test: about 28.7GB / 33.6GB
- Resource numbers stayed stable after the benchmark completed.
- Context checkpoints were created during prompt processing.
- Example checkpoint log entries around 114,688 and 119,763 tokens were about 62.813MiB each.

This is the strongest confirmed no-mmap checkpoint so far.

### 262K Failure

The 262K attempt used the current Q6 split:

```bash
-ngl 99
-ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
-c 262144
```

It failed during context creation:

```text
failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
allocating 812.28 MiB on device 0: cudaMalloc failed: out of memory
failed to allocate compute pp buffers
```

Interpretation:

- The 8-expert split plus `-ngl 99` works at 130K.
- At 262K, larger KV/compute needs require more VRAM headroom.
- llama.cpp detected this during fit, but could not reduce GPU placement because `-ngl 99` was explicitly set.
- Do not remove `-ngl 99` as the default next move; that changes the placement strategy too much.
- The next controlled move is to increase context in steps, then reduce GPU expert layers if a step fails.

### Prompt Cache and Parallel Forking

Sequential same-prefix cache behavior is good:

- Cold 35K base: 35,345 prompt tokens, 1,024 completion tokens, wall 200,150ms.
- Follow-up state: 35,378 prompt tokens, 256 completion tokens, wall 15,453ms.
- Follow-up errors: 35,375 prompt tokens, 256 completion tokens, wall 15,791ms.
- Logs showed LCP similarity, checkpoint restore around 34,829 tokens, and only about 546-549 prompt tokens reprocessed.

Parallel same-prefix forking is not currently good:

- With `-np 2 -c 81920 --cache-ram 4096`, one request reused the hot slot, while the other landed on an empty slot and cold-processed the full 35K prompt.
- `--cache-idle-slots` requires `--kv-unified`; without unified KV it was disabled.
- Even with the unified-KV test direction, logs still showed the second slot cold-prefilling rather than cloning the active hot slot.
- Conclusion: normal llama-server caching is useful for sequential follow-ups, but it does not provide cheap active conversation forking into multiple concurrent slots.
- Related online discussion confirms this is a known gap; true fanout likely needs a llama.cpp patch/API that clones one processed prefix into multiple slots with different suffixes.

### Operational Notes

- Use named `systemd-run --user --scope` scopes for cleanup.
- Ctrl+C can detach from the `systemd-run` client while the scope continues.
- Stop scopes with `systemctl --user kill --kill-whom=all --signal=KILL <scope>.scope`.
- The shell wrapper `bash -lc 'ulimit -c 0; exec "$@"' bash ...` reduces normal core dumping, but does not fully prevent `systemd-coredump` from appearing after some hard crashes.
- If a root-owned `systemd-coredump` worker remains and consumes memory, kill it manually with sudo.

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
9. ~~Consider testing Qwen3.5-35B-A3B Q8_0~~ NOT WORTH IT. IQ4_NL (17.8GB) already OOMs with regex at 188K. Higher quants (Q5=26GB, Q8=37GB) would force -cmoe only (15 t/s), losing the regex fast path. The quality-per-speed tradeoff gets worse, not better. IQ4_NL is the sweet spot for 12GB VRAM.
10. Test reasoning ON for 35B-A3B (the winner) — research how to use thinking tokens effectively with large context, what max_tokens budget is needed, quality impact vs reasoning off. Internet research task.
11. After harness setup: find use cases for 0.8B/2B/4B parallel (300+ t/s aggregate). These models failed at code review but may excel at: autocomplete, inline suggestions, structured extraction, JSON formatting, simple tool routing, draft generation for larger model to refine. Research what the LocalLLaMA community uses sub-4B models for in agentic pipelines.
12. Update HTML wiki at research.visar.veron3.space with real benchmark data

## Directory Structure

- ~/Documents/work/ai/local-llm/ — self-contained
- models/ — GGUF files (12 models downloaded, ~80GB total)
- engine/stable/llama.cpp/ — tracks ggml-org/llama.cpp master, auto-rebuilt daily at 3:30am
- engine/dev/llama.cpp/ — VisarDomi/llama.cpp fork, cherry-picked PRs (branch: dev-prs)
- benchmarks/ — test files, run-test.sh, results/, stats/
- harness/ — Pi (set up), Gastown, OpenCode (not yet set up)
- decisions.md — this file, single source of truth

## Engine Build Commands

```bash
# Stable — full rebuild (also runs daily via llm-stable-build.timer at 3:30am)
cmake -S ~/Documents/work/ai/local-llm/engine/stable/llama.cpp \
      -B ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build --config Release -j$(nproc)

# Dev — full rebuild
cmake -S ~/Documents/work/ai/local-llm/engine/dev/llama.cpp \
      -B ~/Documents/work/ai/local-llm/engine/dev/llama.cpp/build \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build ~/Documents/work/ai/local-llm/engine/dev/llama.cpp/build --config Release -j$(nproc)

# Dev — incremental rebuild (after cherry-picking a PR, only recompiles changed files)
cmake --build ~/Documents/work/ai/local-llm/engine/dev/llama.cpp/build --config Release -j$(nproc)

# Dev — cherry-pick a PR
cd ~/Documents/work/ai/local-llm/engine/dev/llama.cpp
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git checkout dev-prs
git merge pr-<PR_NUMBER>
# resolve conflicts if any, then rebuild
```

## Engine Selection

- `llm switch interactive` — uses stable engine (default)
- `llm switch interactive --dev` — uses dev engine
- `llm status` — shows which engine is running
