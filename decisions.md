# Decisions

## 2026-05-18: Default Qwen3.6 Q6 Workhorse

This repo now defaults to one operational local model:

```text
Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
```

The target use case is sequential coding-agent work on RTX 3060 12GB + 32GB RAM: one active server slot, no mmap/NVMe model paging, fast same-prefix follow-ups, and enough memory headroom for Pi/Codex harness traffic.

Start it:

```bash
llm start
```

Stop it:

```bash
llm stop
```

### Final Operational Shape

```text
CTX=131072
CACHE_RAM=512
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=16
MemoryHigh=28672M
MemoryMax=30720M
MemorySwapMax=0
systemd unit=qwen36-q6-maxctx.scope
proxy unit=go-llm-proxy.service
client endpoint=http://127.0.0.1:8110/v1
KV=f16/default
parallel slots=1
no mmap
reasoning off
```

Expert placement:

```bash
-ngl 99
-ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
```

Meaning:

- GPU expert layers: `0,1,2,10,11,12,20,21`
- CPU expert layers: `3-9,13-19,22-39`

Rationale:

- The earlier 262K full-context profile was validated in controlled benchmarks but real Pi/Codex harness traffic could push it into cgroup memory pressure and collapse prompt processing.
- Halving context to 131K freed enough headroom to move the practical 8-expert split back onto GPU while keeping `--no-mmap`, f16 KV, and the safer `512/65536/16` prompt-cache shape.
- This makes the default faster and more operationally stable for harness use. The 262K profile remains a reference profile for explicit full-context tests.
- `CTX=131072` with 9 GPU experts OOMs; 8 GPU experts is the practical ceiling for this Q6 setup on RTX 3060 12GB.
- Pi harness usage is stable on the 131K/8-expert shape, including tool-cancel flows that previously broke or stalled under the 262K Q6 setup.
- `1024/32768/32` was directly validated and preserved the same useful near-tail cache hit as larger cache configs.
- `1024/65536/32` was selected after the controlled benchmark because it should create about 4 regular checkpoints across 262K instead of 8, reducing checkpoint-copy pressure and memory use while still allowing llama.cpp's near-tail checkpoints for sequential follow-ups.
- Real Pi/Codex harness traffic later showed `1024/65536/32` could exceed `MemoryHigh` at only about 53K live context because repeated tool turns/checkpoints pushed the cgroup into `mem_cgroup_handle_over_high`; the default was therefore lowered to `512/65536/16` for operational headroom.
- `512/65536/16` is now the default operational cache/checkpoint shape.

### Validated Default Run: `131K / 8 GPU Experts / 512/65536/16`

Command shape:

```text
CTX=131072
CACHE_RAM=512
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=16
MemoryHigh=28672M
MemoryMax=30720M
MemorySwapMax=0
GPU experts=0,1,2,10,11,12,20,21
```

Base 120K benchmark:

- Result file: `benchmarks/results/20260518-160940-qwen3.6-q6-131072-120k.json`
- Prompt tokens: `119767`
- Completion tokens: `1024`
- Prompt eval: `563,810.34 ms`, `212.42 t/s`
- Generation eval: `51,067.45 ms`, `20.05 t/s`
- Total: `614,877.80 ms`
- Wall: `615,234 ms`
- Tail checkpoints: `119251` and `119763`.

Same-prefix follow-up:

- Result file: `benchmarks/results/20260518-161015-qwen3.6-q6-131072-120k-followup.json`
- Prompt tokens: `119799`
- Completion tokens: `512`
- Restored checkpoint: `119251`
- Reprocessed prompt tokens: `548`
- Prompt eval: `3,759.33 ms`, `145.77 t/s`
- Generation eval: `25,373.84 ms`, `20.18 t/s`
- Total: `29,133.17 ms`
- Wall: `29,471 ms`

Pi/Codex harness validation: stable enough to keep as default; tool cancellations no longer break the flow in the observed test.

### Full-Context Reference Run: `262K / 3 GPU Experts / 1024/32768/32`

Command shape:

```text
CTX=262144
CACHE_RAM=1024
CHECKPOINT_EVERY=32768
CTX_CHECKPOINTS=32
MemoryHigh=28672M
MemoryMax=30720M
MemorySwapMax=0
```

Base 241K benchmark:

- Result file: `benchmarks/results/20260516-004300-qwen3.6-q6-262144-241k.json`
- Prompt tokens: `240551`
- Completion tokens: `1024`
- Prompt eval: `1,340,387.18 ms`, `179.46 t/s`
- Generation eval: `66,287.63 ms`, `15.45 t/s`
- Total: `1,406,674.81 ms`
- Checkpoints: `9/32` by the 241K prompt.
- Tail checkpoints: `240035` and `240547`.

Same-prefix follow-up:

- Result file: `benchmarks/results/20260516-004344-qwen3.6-q6-262144-241k-followup.json`
- API `cached_tokens`: `240035`
- Reprocessed prompt tokens: `548`
- Prompt eval: `4,593.17 ms`, `119.31 t/s`
- Generation eval: `32,729.32 ms`, `15.64 t/s`
- Total: `37,322.48 ms`

Resource reading after this run:

- System RAM: about `30GiB / 31GiB` used, `1.1GiB` available.
- System swap: about `1.9GiB / 31GiB` used globally.
- llama cgroup swap: `0`.
- Scope current memory: `29,511,843,840` bytes (`27.5GiB`).
- Scope peak memory: `30,040,125,440` bytes (`28.0GiB`).
- Scope headroom before `MemoryMax`: `552,927,232` bytes (`527MiB`).
- Scope guard: `MemoryHigh=30,064,771,072` bytes (`28.0GiB`), `MemoryMax=32,212,254,720` bytes (`30.0GiB`).
- VRAM: `11029MiB / 12288MiB` used, `880MiB` free.

### Why Tail Checkpoints Matter

Regular checkpoint math is only the floor. llama.cpp also creates near-end checkpoints when slots are available. For sequential agentic coding, follow-ups happen at the conversation tail, so tail checkpoints matter more than dense mid-context coverage.

Observed behavior:

- `2048/8192/32`: reached `31/32` by 241K and restored from `240035`.
- `2048/16384/64`: reached `16/64` by 241K and restored from `240035`.
- `1024/32768/32`: reached `9/32` by 241K and restored from `240035`.

All three reprocessed only `548` prompt tokens on the same follow-up. For the 262K reference profile, this favors fewer regular checkpoints and lower memory pressure over dense mid-context checkpoint coverage.

### Useful Comparisons

| Metric | `2048/16384/64` | `1024/32768/32` |
|---|---:|---:|
| Checkpoints by 241K | `16/64` | `9/32` |
| Tail checkpoint restored | `240035` | `240035` |
| Follow-up reprocessed | `548` | `548` |
| Base PP | `179.57 t/s` | `179.46 t/s` |
| Base TG | `15.80 t/s` | `15.45 t/s` |
| Base total | `1,404,375.12 ms` | `1,406,674.81 ms` |
| Follow-up PP | `118.44 t/s` | `119.31 t/s` |
| Follow-up TG | `15.81 t/s` | `15.64 t/s` |
| Follow-up total | `37,007.13 ms` | `37,322.48 ms` |
| System RAM available | `918MiB` | `1.1GiB` |
| Scope headroom | `159MiB` | `527MiB` |
| VRAM free | `790MiB` | `880MiB` |

The `1024/32768/32` shape is slightly slower in total time but materially lighter than the larger-cache 262K alternatives. It remains the best recorded 262K reference run, but the operational default is now the 131K 8-expert profile above.

## Historical Boundaries

### Q8

`Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` is not practical with `--no-mmap` on this machine:

- mmap + regex could load and answer at 262K.
- no-mmap was either RAM-bound or VRAM-bound depending on expert placement.
- 10 GPU expert layers reached about `11.6GB` VRAM then failed on an additional CUDA compute buffer.
- Conclusion: Q8 no-mmap is not viable on 32GB RAM + 12GB VRAM.

### Superseded 130K Quality Mode

The earlier 8-GPU-expert split proved that 130K context could carry more experts on GPU:

```text
CTX=131072
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
GPU experts: 0,1,2,10,11,12,20,21
```

120K benchmark:

- Prompt tokens: `119767`
- Completion tokens: `1024`
- Prompt eval: `552,143.42 ms`, `216.91 t/s`
- Generation eval: `51,056.51 ms`, `20.06 t/s`
- Follow-up restored near `119251` and reprocessed `548` prompt tokens.

This is faster, but it is not full-context.

This was superseded by the default `131K / 8 GPU experts / 512/65536/16` run above, which keeps the same expert split with lower prompt-cache pressure.

### Parallel Forking

Normal llama-server caching works well for sequential same-prefix follow-ups. It does not cheaply fork one hot conversation into multiple concurrent slots:

- With `-np 2`, one request reused the hot slot while the other landed on an empty slot and cold-prefilled.
- `--cache-idle-slots` requires unified KV, and tests still did not produce cheap active-slot cloning.
- True parallel branch fanout likely needs a llama.cpp patch/API that clones one processed prefix into multiple slots.

## Operational Notes

- `--no-mmap` is part of the default. The goal is no NVMe-backed model paging.
- Keep OS swap available for the desktop, but keep llama's cgroup swap disabled with `MemorySwapMax=0`.
- The `llm` script starts the server under the same user systemd scope shape used during manual testing.
- `llm start` starts `go-llm-proxy.service` on `127.0.0.1:8110`, then sources `~/.config/cuda-env.sh`, runs llama-server through `systemd-run --user --scope`, wraps it with `timeout 60m`, disables core dumps with `ulimit -c 0`, and binds llama-server to `0.0.0.0:8100`.
- Codex should target `http://127.0.0.1:8110/v1`, not raw llama-server. The proxy is configured with `responses_mode: translate`, which translates Codex Responses API traffic into Chat Completions for llama-server and drops unsupported non-function Responses items before they hit llama.cpp.
- Operational flow is `llm start` and `llm stop`.
- `llm stop` is the unload path for both the proxy and llama-server.
- CUDA 13.1.2 is the intended toolkit for this llama.cpp path; source `~/.config/cuda-env.sh` for manual testing.
