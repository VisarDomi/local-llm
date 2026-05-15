# Current Test: Qwen3.6 Q6 Agent Coding

This file is only the active copy-paste runbook. Historical results belong in `decisions.md`.

Goal: run `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` as a single agent-coding server on RTX 3060 12GB + 32GB RAM.

Current baseline:

- model: `models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`
- server: `engine/stable/llama.cpp/build/bin/llama-server`
- placement: `-ngl 99` plus regex CPU expert override
- parallelism: disabled with `-np 1`
- prompt cache: full-context default is `--cache-ram 512`
- checkpoint spacing: full-context default is `--checkpoint-every-n-tokens 32768`
- mmap: disabled with `--no-mmap`
- KV: default f16 for quality. Add Q8 KV only for long-context mode.
- memory guard: full-context default is `MemoryHigh=29696M`, `MemoryMax=31744M`, `MemorySwapMax=0`

The older exploratory guard `MemoryHigh=30720M`, `MemoryMax=32768M` allowed the desktop to run out of breathing room during a full-context run. The safer `28672M/30720M` guard was too tight for full context. Use `29696M/31744M` for the current workhorse.

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

Fixed:

```text
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
```

Known-good:

```text
CTX=131072
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
OT_REGEX="blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

CTX=163840
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
OT_REGEX="blk\.([2-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"

CTX=262144
CACHE_RAM=512
CHECKPOINT_EVERY=32768
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

Confirmed winner: 3 GPU experts loads full context with `CTX=262144`, `CACHE_RAM=512`, `CHECKPOINT_EVERY=32768`, and `MemoryHigh=29696M` / `MemoryMax=31744M`.

Suggested order for higher contexts: try the next context with 7 GPU experts first, then 6, then 5.

## Start Server

```bash
cd ~/Documents/work/ai/local-llm
source ~/.config/cuda-env.sh
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-maxctx.scope

CTX=262144
CACHE_RAM=1024
CHECKPOINT_EVERY=16384
OT_REGEX="blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU"
systemd-run --user --scope \
  --expand-environment=no \
  --unit=qwen36-q6-maxctx \
  -p KillSignal=SIGKILL \
  -p SendSIGKILL=yes \
  -p TimeoutStopSec=2s \
  -p MemoryMax=32256M \
  -p MemoryHigh=31232M \
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
