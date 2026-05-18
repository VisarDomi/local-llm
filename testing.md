# Current Test: Qwen3.6 Q6 Agent Coding

This file is only the active copy-paste runbook. Historical results belong in `decisions.md`.

Goal: run `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` as a single agent-coding server on RTX 3060 12GB + 32GB RAM.

Current baseline:

- model: `models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`
- server: `engine/stable/llama.cpp/build/bin/llama-server`
- placement: `-ngl 99` plus regex CPU expert override
- parallelism: disabled with `-np 1`
- prompt cache: full-context default is `--cache-ram 2048`
- checkpoint spacing: full-context default is `--checkpoint-every-n-tokens 16384`
- context checkpoints: full-context default is `--ctx-checkpoints 64`
- mmap: disabled with `--no-mmap`
- KV: default f16 for quality. Add Q8 KV only for long-context mode.
- memory guard: full-context default is `MemoryHigh=28672M`, `MemoryMax=30720M`, `MemorySwapMax=0`

The older exploratory guard `MemoryHigh=30720M`, `MemoryMax=32768M` allowed the desktop to run out of breathing room during a full-context run. The aggressive `31232M/32256M` guard fit the `2048/8192/32` test but made the desktop feel heavy. The current test direction lowers the guard back to `28672M/30720M` and reduces regular checkpoint pressure. Keep swap enabled at the OS level, but keep llama's cgroup swap disabled with `MemorySwapMax=0`.

## Cleanup

Use this if Ctrl+C leaves a scope alive:

```bash
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-maxctx.scope 2>/dev/null || true
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-35k-kvu.scope 2>/dev/null || true
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-35k-parallel.scope 2>/dev/null || true
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-cache.scope 2>/dev/null || true
pgrep -af 'llama-server|llama-cli|Qwen3.6|systemd-coredump|coredump' || true
free -h
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
```

If `systemd-coredump` is still running and eating memory:

```bash
pgrep -af 'systemd-coredump|coredump' || true
sudo pkill -KILL -f '/usr/lib/systemd/systemd-coredump'
pgrep -af 'systemd-coredump|coredump' || true
free -h
```

## Current Variables

Harness-pressure baseline:

```text
CTX=131072
CACHE_RAM=512
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=16
MemoryHigh=28672M
MemoryMax=30720M
```

Known-good:

```text
CTX=131072
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
CTX_CHECKPOINTS=32
OT_REGEX="blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

CTX=163840
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
CTX_CHECKPOINTS=32
OT_REGEX="blk\.([2-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

CTX=262144
CACHE_RAM=2048
CHECKPOINT_EVERY=8192
CTX_CHECKPOINTS=32
OT_REGEX="blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
```

Next context steps:

```text
147456
163840
180224
196608
212992
229376
245760
262144  # model max context
```

## Regex Steps

Current split keeps these expert layers on GPU:

```text
0,1,2,10,11,12,20,21
```

Use these `OT_REGEX` values to remove one GPU expert layer at a time. Each step moves one more expert layer to CPU and frees more VRAM, at the cost of some generation speed.

```bash
# 9 GPU expert layers: 0,1,2,10,11,12,20,21,22
OT_REGEX="blk\.([3-9]|1[3-9]|2[3-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 8 GPU expert layers: 0,1,2,10,11,12,20,21
OT_REGEX="blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 7 GPU expert layers: 0,1,10,11,12,20,21
OT_REGEX="blk\.([2-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 6 GPU expert layers: 0,1,10,11,20,21
OT_REGEX="blk\.([2-9]|1[2-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 5 GPU expert layers: 0,1,10,11,20
OT_REGEX="blk\.([2-9]|1[2-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 4 GPU expert layers: 0,1,10,20
OT_REGEX="blk\.([2-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 3 GPU expert layers: 0,10,20
OT_REGEX="blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 2 GPU expert layers: 0,10
OT_REGEX="blk\.([1-9]|1[1-9]|2[0-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

# 1 GPU expert layer: 0
OT_REGEX="blk\.([1-9]|[1-3][0-9])\.ffn_.*_exps\.weight=CPU"

# 0 GPU expert layers: all experts on CPU
OT_REGEX="blk\.([0-9]|[1-3][0-9])\.ffn_.*_exps\.weight=CPU"
```

Confirmed: 7 GPU experts loads with `CTX=163840`.

Current operational winner: 8 GPU experts loads and passes the 120K base/follow-up benchmarks with `CTX=131072`, `CACHE_RAM=512`, `CHECKPOINT_EVERY=65536`, `CTX_CHECKPOINTS=16`, and `MemoryHigh=28672M` / `MemoryMax=30720M`.

Full-context reference: 3 GPU experts loads full context with `CTX=262144`, `CACHE_RAM=1024`, `CHECKPOINT_EVERY=32768`, `CTX_CHECKPOINTS=32`, and `MemoryHigh=28672M` / `MemoryMax=30720M`.

Current hypothesis: lower context should free enough RAM/VRAM headroom to move more experts back to GPU. Expert-load failures usually show up during model load or warmup, so test expert count first with a smoke prompt before running a 20 minute context benchmark.

Suggested low-context order:

```text
CTX=131072, 8 GPU experts, CACHE_RAM=512, CHECKPOINT_EVERY=65536, CTX_CHECKPOINTS=16
CTX=160000, 8 GPU experts, CACHE_RAM=512, CHECKPOINT_EVERY=65536, CTX_CHECKPOINTS=16
CTX=160000, 7 GPU experts, CACHE_RAM=512, CHECKPOINT_EVERY=65536, CTX_CHECKPOINTS=16
```

Rejected: `CTX=131072` with 9 GPU experts OOMs. Treat 8 GPU experts as the practical 131K ceiling for Q6 on this 12GB card.

Suggested order for higher contexts: try the next context with 7 GPU experts first, then 6, then 5.

Checkpoint note: regular coverage math is only the floor. llama.cpp also creates near-end checkpoints when slots are available, so same-prefix follow-ups near the end of a long prompt can reprocess only a small tail. Example: the 241K full-context run restored from 240,035 tokens and reprocessed 548 prompt tokens, not half of `CHECKPOINT_EVERY`.

Source-of-truth defaults in this llama.cpp build are `--cache-ram 8192`, `--checkpoint-every-n-tokens 8192`, and `--ctx-checkpoints 32`. For this single-slot 262K setup, `2048/8192/32` is the practical cache shape: 32 checkpoints * 8192 tokens covers 262,144 tokens, and 32 checkpoints * 62.813MiB is about 2010MiB. Raising `--ctx-checkpoints` above 32 only helps if `CACHE_RAM` is raised enough to hold the extra checkpoints.

## Cache/Checkpoint Test Matrix

Use the safer memory guard for these:

```text
MemoryHigh=28672M
MemoryMax=30720M
MemorySwapMax=0
```

The first line in each pair is the requested cache size. The second line doubles `CHECKPOINT_EVERY` to reduce regular checkpoint creation and memory pressure while keeping extra checkpoint slots available for tail checkpoints.

```text
# A: practical sequential-agent candidate
CACHE_RAM=2048
CHECKPOINT_EVERY=16384
CTX_CHECKPOINTS=64

# A2: lighter regular checkpoint pressure
CACHE_RAM=2048
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=64

# B: high cache, default spacing, more checkpoint slots
CACHE_RAM=4096
CHECKPOINT_EVERY=8192
CTX_CHECKPOINTS=64

# B2: lower regular checkpoint pressure
CACHE_RAM=4096
CHECKPOINT_EVERY=16384
CTX_CHECKPOINTS=64

# C: high cache, coarser spacing, many checkpoint slots
CACHE_RAM=4096
CHECKPOINT_EVERY=16384
CTX_CHECKPOINTS=128

# C2: lower regular checkpoint pressure
CACHE_RAM=4096
CHECKPOINT_EVERY=32768
CTX_CHECKPOINTS=128

# D: default cache-ram budget, coarse spacing
CACHE_RAM=8192
CHECKPOINT_EVERY=32768
CTX_CHECKPOINTS=128

# D2: lower regular checkpoint pressure
CACHE_RAM=8192
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=128
```

Expected feel: A should be smoother than `2048/8192/32` because it halves regular checkpoint creation while leaving tail slots. A2 should be smoother still, but mid-context restore granularity is worse. B/C/D give llama permission to hold much more prompt-cache RAM; they may still feel heavy even if regular checkpoint spacing is coarser.

## Start Server

```bash
cd ~/Documents/work/ai/local-llm
source ~/.config/cuda-env.sh
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-maxctx.scope

CTX=131072
CACHE_RAM=512
CHECKPOINT_EVERY=32768
CTX_CHECKPOINTS=16
OT_REGEX="blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
systemd-run --user --scope \
  --expand-environment=no \
  --unit=qwen36-q6-maxctx \
  -p KillSignal=SIGKILL \
  -p SendSIGKILL=yes \
  -p TimeoutStopSec=2s \
  -p MemoryMax=30720M \
  -p MemoryHigh=28672M \
  -p MemorySwapMax=0 \
  timeout 60m \
  bash -lc 'ulimit -c 0; exec "$@"' bash \
  ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
    -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
    --port 8100 \
    --host 127.0.0.1 \
    -ngl 99 \
    -ot "$OT_REGEX" \
    --no-mmap \
    -np 1 \
    --cache-ram "$CACHE_RAM" \
    --checkpoint-every-n-tokens "$CHECKPOINT_EVERY" \
    --ctx-checkpoints "$CTX_CHECKPOINTS" \
    --no-cache-idle-slots \
    -c "$CTX" \
    --jinja \
    --reasoning off
```

## Smoke Test

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "Answer in one short sentence: what is 2+2?"}
    ],
    "max_tokens": 16
  }'
```

## Inspect State

```bash
curl -sf http://127.0.0.1:8100/health
curl -sf http://127.0.0.1:8100/slots | jq '.[] | {id,n_ctx,is_processing}'
free -h
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
```

## Run Benchmarks

Run the base file first, then the matching follow-up file without restarting the server. The follow-up is a same-prefix cache probe.

```bash
LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-35k.json \
  --tag qwen3.6-q6-40000-35k

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-35k-followup-ownership.json \
  --tag qwen3.6-q6-40000-35k-followup

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-120k.json \
  --tag qwen3.6-q6-131072-120k

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-120k-followup-ownership.json \
  --tag qwen3.6-q6-131072-120k-followup

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-177k.json \
  --tag qwen3.6-q6-180224-177k

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-177k-followup-ownership.json \
  --tag qwen3.6-q6-180224-177k-followup

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-211k.json \
  --tag qwen3.6-q6-212992-211k

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-211k-followup-ownership.json \
  --tag qwen3.6-q6-212992-211k-followup

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-241k.json \
  --tag qwen3.6-q6-262144-241k

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-241k-followup-ownership.json \
  --tag qwen3.6-q6-262144-241k-followup
```

## Stop Between Rounds

```bash
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-maxctx.scope
systemctl --user reset-failed qwen36-q6-maxctx.scope 2>/dev/null || true
pgrep -af 'llama-server|llama-cli|Qwen3.6|systemd-coredump|coredump' || true
free -h
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
```

## Long-Context Mode

For lower KV quality but more context headroom, add Q8 KV:

```bash
--cache-type-k q8_0 \
--cache-type-v q8_0
```

Known Q8 KV checkpoint:

- `-c 131072`
- `benchmarks/context-scale-120k.json`
- Prompt: 119,767 tokens at 217.05 tokens/s
- Generation: 1,024 tokens at 16.75 tokens/s

## If VRAM Fails

Retry the same context with the next `OT_REGEX` from **Regex Steps** before changing cache or KV.
