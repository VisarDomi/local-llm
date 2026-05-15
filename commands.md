# Commands

## Qwen3.6 35B Q6 XL no-mmap tests

Model:

```bash
~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
```

Use the CUDA 13.1 environment first:

```bash
source ~/.config/cuda-env.sh
```

### Tested working: 4K context, no mmap

This loaded and answered a simple prompt on 2026-05-15. It keeps expert tensors on GPU for layers `0,1,2,10,11,12,20,21` and sends expert tensors for layers `3-9,13-19,22-39` to CPU.

```bash
systemd-run --user --scope \
  -p MemoryMax=29184M \
  -p MemoryHigh=27136M \
  -p MemorySwapMax=0 \
  timeout 6m \
  ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-cli \
    -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
    -ngl 99 \
    -ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
    --no-mmap \
    -c 4096 \
    --jinja \
    --reasoning off \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    -p "Answer in one short sentence: what is 2+2?" \
    -n 16
```

Observed result:

- Answered `2+2 equals 4.`
- Prompt: 48.4 t/s
- Generation: 24.0 t/s
- Cgroup memory: about 25.9GB
- Process RSS: about 23.8GB
- VRAM: about 9.0GB used
- Global available RAM: about 5.9GB
- Swap was already dirty from earlier tests, but this scoped run used `MemorySwapMax=0`.

Next scaling step:

- Keep this split and try larger contexts first: `262144`.
- If context OOMs in VRAM, reduce KV pressure before changing the model split.
- If RAM becomes the problem, try one more GPU-heavy split before giving up: keep expert layers `0,1,2,3,10,11,12,20,21` on GPU and send `4-9,13-19,22-39` to CPU.

### Safe systemd-run pattern

Use a named scope and disable core dumps inside the shell wrapper. `Ctrl+C` only cancels the foreground `systemd-run` client; the scope can keep running. Stop by killing the named scope from another terminal.

```bash
systemctl --user reset-failed qwen36-q6-test.scope 2>/dev/null || true

systemd-run --user --scope \
  --unit=qwen36-q6-test \
  -p KillSignal=SIGKILL \
  -p SendSIGKILL=yes \
  -p TimeoutStopSec=2s \
  -p MemoryMax=32768M \
  -p MemoryHigh=30720M \
  -p MemorySwapMax=0 \
  timeout 60m \
  bash -lc 'ulimit -c 0; exec "$@"' bash \
  ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
    -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
    --port 8100 \
    --host 127.0.0.1 \
    -ngl 99 \
    -ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
    --no-mmap \
    -np 1 \
    --cache-ram 0 \
    --no-cache-idle-slots \
    -c 262144 \
    --jinja \
    --reasoning off \
    --cache-type-k q8_0 \
    --cache-type-v q8_0
```

Instant cancel:

```bash
systemctl --user kill --kill-whom=all --signal=KILL qwen36-q6-test.scope
```

Verify cleanup:

```bash
systemctl --user list-units --type=scope --all --no-pager | rg 'qwen36|llama|Qwen' || true
pgrep -af 'llama-server|llama-cli|Qwen3.6|systemd-coredump' || true
free -h
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
```

### Tested working: 40K context, 35K benchmark

This loaded as `llama-server` with one slot, prompt cache disabled, no mmap, and the same 8-expert GPU split.

```bash
systemd-run --user --scope \
  -p MemoryMax=29184M \
  -p MemoryHigh=27136M \
  -p MemorySwapMax=0 \
  timeout 25m \
  ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
    -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
    --port 8100 \
    --host 127.0.0.1 \
    -ngl 99 \
    -ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
    --no-mmap \
    -np 1 \
    --cache-ram 0 \
    --no-cache-idle-slots \
    -c 40960 \
    --jinja \
    --reasoning off \
    --cache-type-k q8_0 \
    --cache-type-v q8_0
```

Benchmark:

```bash
LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-35k.json \
  --tag qwen3.6-35b-q6-40k-nommap-35k
```

Observed result:

- Result file: `benchmarks/results/20260515-133923-qwen3.6-35b-q6-40k-nommap-35k.json`
- Prompt tokens: 35,345
- Completion tokens: 1,024
- Finish reason: `length`
- Wall time: 202,022ms
- Approx total throughput from wall: 180 tokens/s across prompt plus completion
- Server loaded in about 60 seconds.
- Before benchmark: cgroup memory about 26.4GB, VRAM about 9.4GB, global available RAM about 5.8GB.
- During benchmark: resource use stayed stable around 9.4GB VRAM and 5.4GB available RAM.
- Tiny-prompt PP numbers are misleading; the 4K smoke prompt reported 48.4 prompt t/s, but the 35K benchmark completed far faster than that would imply.

### Tested working: 130K context, 120K benchmark

This loaded as `llama-server` with one slot, prompt cache disabled, no mmap, and the same 8-expert GPU split. This is the exact server command used for the successful 120K benchmark.

```bash
source ~/.config/cuda-env.sh

systemd-run --user --scope \
  -p MemoryMax=30720M \
  -p MemoryHigh=28672M \
  -p MemorySwapMax=0 \
  timeout 30m \
  ~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
    -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
    --port 8100 \
    --host 127.0.0.1 \
    -ngl 99 \
    -ot "blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
    --no-mmap \
    -np 1 \
    --cache-ram 0 \
    --no-cache-idle-slots \
    -c 131072 \
    --jinja \
    --reasoning off \
    --cache-type-k q8_0 \
    --cache-type-v q8_0
```

Benchmark:

```bash
cd ~/Documents/work/ai/local-llm

LLM_URL=http://localhost:8100/v1/chat/completions \
  benchmarks/run-test.sh benchmarks/context-scale-120k.json \
  --tag qwen3.6-35b-q6-130k-nommap-120k
```

Observed result:

- Prompt tokens processed: 119,767
- Completion tokens: 1,024
- Truncated: 0
- Prompt eval: 551,801.48ms, 217.05 tokens/s
- Generation eval: 61,130.88ms, 16.75 tokens/s
- Total time: 612,932.36ms for 120,791 tokens
- Live VRAM after load/test: about 10,775MiB / 12,288MiB
- Live RAM after load/test: about 28.7GB / 33.6GB
- Resource numbers stayed stable after the benchmark completed.
- Context checkpoints were created during processing; examples in the log show checkpoints around 114,688 and 119,763 tokens, each about 62.813MiB.

## Qwen3.6 35B Q8 XL max-context tests

Model:

```bash
~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
```

Use the CUDA 13.1 environment first:

```bash
source ~/.config/cuda-env.sh
```

### Tested working: 262K context with mmap

This loaded and answered a simple prompt on 2026-05-15. It uses mmap, so it may page from NVMe. The split keeps expert tensors on GPU only for layers `0,10,20` and sends expert tensors for layers `1-9,11-19,21-39` to CPU.

```bash
~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
  -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --port 8100 \
  --host 127.0.0.1 \
  -ngl 99 \
  -ot "blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
  --jinja \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -c 262144
```

Simple CLI smoke test equivalent:

```bash
~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-cli \
  -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  -ngl 99 \
  -ot "blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
  -c 262144 \
  --jinja \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -p "Answer in one short sentence: what is 2+2?" \
  -n 16
```

### No-mmap target

This is the next target for avoiding lazy NVMe paging. It forces a single server slot and disables the server prompt cache so the test is about model + KV residency, not four concurrent 262K slots or idle-slot snapshots.

```bash
~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
  -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --port 8100 \
  --host 127.0.0.1 \
  -ngl 99 \
  -ot "blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU" \
  --no-mmap \
  -np 1 \
  --cache-ram 0 \
  --no-cache-idle-slots \
  --jinja \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -c 262144
```

### Baseline: current `llm` max-context shape

This mirrors the existing `llm switch workhorse-full` shape: full 262K context, Q8 KV, Jinja chat template, reasoning disabled, and all MoE experts on CPU via `-cmoe`. For this Q8 model it loads too much on the CPU/RAM side and was worse than selective regex during smoke testing.

```bash
~/Documents/work/ai/local-llm/engine/stable/llama.cpp/build/bin/llama-server \
  -m ~/Documents/work/ai/local-llm/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --port 8100 \
  --host 127.0.0.1 \
  -ngl 99 \
  -cmoe \
  --no-mmap \
  --jinja \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -c 262144
```

Notes:

- No-mmap 4K smoke test with the three-expert-layer split was stopped on 2026-05-15 before completion: it reached about 22.3GB cgroup memory plus 1.9GB swap, while gallery OCR was also using GPU memory. This split does not meet the "leave 6-8GB RAM for the system" requirement.
- Clean no-mmap 4K fit probe after stopping gallery OCR also failed the fit test: with `MemoryMax=22G`, `MemoryHigh=20G`, and `MemorySwapMax=0`, it reached about 20.1GB cgroup memory and had not finished loading. Global swap still rose to about 627MiB, so the test was stopped.
- Five-expert-layer no-mmap split, keeping layers `0,1,10,11,20` on GPU, reached about 22.6GB cgroup memory and 8.3GB VRAM before loading; stopped at the high boundary.
- Eight-expert-layer no-mmap split, keeping layers `0,1,2,10,11,12,20,21` on GPU, reached about 26.6GB cgroup memory, 27.9GB RSS, 9.9GB VRAM, 2.3GB system-available RAM, and still had not loaded; stopped near freeze territory.
- Twelve-expert-layer no-mmap split, keeping layers `0-3,10-13,20-23` on GPU, failed immediately with CUDA OOM: attempted a 12576.11MiB CUDA allocation.
- Ten-expert-layer no-mmap split, keeping layers `0-3,10-12,20-22` on GPU, reached about 25.5GB cgroup memory, 26.6GB RSS, and 11.6GB VRAM, then failed to allocate an additional 568.28MiB CUDA compute buffer and stalled. This is the observed tight boundary: RAM and VRAM are both nearly full and it still does not load.
- Run future no-mmap tests under a cgroup guard, for example `systemd-run --user --scope -p MemoryMax=22G -p MemoryHigh=20G -p MemorySwapMax=0 timeout 6m ...`.
- Stop `gallery-ocr.service` before VRAM-fit tests if clean GPU headroom matters.
- The working server auto-selected `n_parallel = 4`, created four 262K slots, and enabled an 8192 MiB RAM prompt cache. That is useful for throughput/reuse, but it is hostile to a tight single-user no-mmap fit test.
- The five-expert-layer split, keeping layers `0,1,10,11,20` on GPU, worked at 4K context but failed at 262K with CUDA OOM while allocating a 1057 MiB compute buffer.
- The three-expert-layer split, keeping layers `0,10,20` on GPU, worked at 262K with mmap and answered `2+2 equals 4`.
- Observed 262K working run: about 9.0GB VRAM used and about 24.8GB process RSS after load, with swap full from paging.
- `--no-mmap` is still unproven for this Q8 model. If the single-slot no-cache command fits, try moving more expert layers back to GPU to reduce CPU RAM pressure.
- The tensor names in this file use `.weight`, for example `blk.0.ffn_down_exps.weight`, so the regex uses `ffn_.*_exps\.weight=CPU`.
