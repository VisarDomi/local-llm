# Local LLM

Local llama.cpp workspace for finding a practical coding-agent model on this PC:

- CPU: Xeon E5-1650 v3, 6 physical cores
- RAM: 32GB class system memory
- GPU: RTX 3060 12GB
- Server: `engine/stable/llama.cpp/build/bin/llama-server`

## Default Model

The current default is the Qwen3.6 35B MoE Q6 workhorse:

```text
models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
```

It is tuned for sequential coding-agent use: one active server slot, no mmap/NVMe model paging, full 262K context, f16 KV, and selective MoE expert placement.

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
CTX=262144
CACHE_RAM=1024
CHECKPOINT_EVERY=65536
CTX_CHECKPOINTS=32
MemoryHigh=28672M
MemoryMax=30720M
MemorySwapMax=0
systemd unit=qwen36-q6-maxctx.scope
```

The active expert split keeps only layers `0,10,20` expert tensors on GPU and sends the remaining expert tensors to CPU:

```text
blk\.([1-9]|1[1-9]|2[1-9]|3[0-9])\.ffn_.*_exps\.weight=CPU
```

See `decisions.md` for the measurements and why this is the current default.

`llm start` runs this shape through the same `systemd-run --user --scope`
command used during manual testing, sources `~/.config/cuda-env.sh`, disables
core dumps with `ulimit -c 0`, and binds llama-server to `127.0.0.1:8100`.

## Notes

- `commands.md`, `testing.md`, and `continue.md` were removed after the Qwen3.6 tuning run. Durable conclusions belong in `decisions.md`.
- PaddleOCR no longer uses a repo-local `.venv-paddleocr` symlink here. OCR runtime ownership is outside this repository.
