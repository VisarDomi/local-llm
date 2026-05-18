# Local LLM

Local llama.cpp workspace for finding a practical coding-agent model on this PC:

- CPU: Xeon E5-1650 v3, 6 physical cores
- RAM: 32GB class system memory
- GPU: RTX 3060 12GB
- Server backend: `engine/stable/llama.cpp/build/bin/llama-server`
- Client endpoint: local `go-llm-proxy` on `127.0.0.1:8110`

## Default Model

The current default is the Qwen3.6 35B MoE Q6 workhorse:

```text
models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
```

It is tuned for sequential coding-agent use: one active server slot, no mmap/NVMe model paging, 131K context, f16 KV, and a faster 8-layer MoE expert placement. The old 262K profile remains documented in `decisions.md` as the full-context reference.

Start it:

```bash
llm start
```

Stop it:

```bash
llm stop
```

Check status:

```bash
llm status
```

## Default Shape

The default operational shape is:

```text
CTX=131072
CACHE_RAM=512
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=16
MemoryHigh=28672M
MemoryMax=30720M
MemorySwapMax=0
systemd unit=qwen36-q6-maxctx.scope
```

The active expert split keeps layers `0,1,2,10,11,12,20,21` expert tensors on GPU and sends the remaining expert tensors to CPU:

```text
blk\.([3-9]|1[3-9]|2[2-9]|3[0-9])\.ffn_.*_exps\.weight=CPU
```

See `decisions.md` for the measurements and why this is the current default.

`llm start` runs this shape through the same `systemd-run --user --scope`
command used during manual testing, sources `~/.config/cuda-env.sh`, disables
core dumps with `ulimit -c 0`, and binds llama-server to `0.0.0.0:8100`.

Codex and other OpenAI Responses API clients should use the proxy endpoint:

```text
http://127.0.0.1:8110/v1
```

The proxy forces `responses_mode: translate`, so Codex/Pi can speak Responses
API while llama.cpp receives Chat Completions requests. This avoids depending on
llama.cpp's partial native Responses path for tool-heavy agent traffic.

## Notes

- `testing.md` is the active scratch runbook for context/expert tests. Durable conclusions belong in `decisions.md`.
- `commands.md` and `continue.md` were removed after the Qwen3.6 tuning run.
- PaddleOCR no longer uses a repo-local `.venv-paddleocr` symlink here. OCR runtime ownership is outside this repository.
