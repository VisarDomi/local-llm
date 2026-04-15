# Continue from here

Read decisions.md first — it has benchmark data, architecture decisions, engine build commands, and directory structure.

## What happened this session (session 3)

### Harness setup
1. **Pi Coding Agent** — installed (v0.66.1), configured at `~/.pi/agent/models.json` with single `local/default` provider. Alias `pi` in `~/.bash_aliases`. Works with 9B on chat completions. Known issue: 9B overwrites files instead of using edit tool (known small-model failure mode).
2. **Codex CLI** — installed (v0.120.0), configured at `~/.codex/config.toml`. Default uses OpenAI sub, local provider configured. BUT: Codex only speaks Responses API (`/v1/responses`), not Chat Completions. Needs dev build.
3. **Researched all CLI agents** — found non-React options: Codex (Rust/Ratatui), Crush (Go/Bubbletea), Aider (Python/prompt_toolkit), OpenCode (TS/Zig). Pi is actually custom TUI, not React.

### Infrastructure changes
1. **Unified port**: All tiers now serve on port 8100 (was 8010/8100/8200). Pi/Codex/run-test.sh all point to 8100.
2. **Dual engine**: `engine/stable/` (tracks upstream master) and `engine/dev/` (VisarDomi/llama.cpp fork). `llm switch <tier> --dev` flag selects engine.
3. **Daily auto-build**: `llm-stable-build.timer` at 3:30am pulls master and rebuilds stable. Uses ccache.
4. **run-test.sh**: Auto-detects running server on port 8100 (no more wrong-port errors).
5. **Python venv**: `~/Documents/work/ai/local-llm/engine/venv/` for convert_hf_to_gguf.py deps (torch CPU, transformers, gguf).

### MTP experiment (PR #20700) — FAILED, cleaned up
- Converted Qwen3.5-9B with MTP tensors using dev branch converter
- Used Unsloth's imatrix (`~/Documents/work/ai/local-llm/models/qwen3.5-9b-imatrix-unsloth.gguf` — kept, 5MB)
- Result: "speculative decoding not supported by this context" — Qwen3.5's hybrid SSM architecture breaks it
- Even when server loaded, output was garbage (stuttering repetition), no speed gain
- **Verdict**: Wait for PR to mature and for Unsloth to publish MTP-enabled GGUFs
- All MTP model files deleted (F16, IQ4_NL, HF safetensors)

### Quantization knowledge gained
- **Unsloth secret sauce**: Dynamic per-layer mixed precision + proprietary chat-optimized calibration dataset (not published)
- **mradermacher**: Standard llama.cpp + wiki imatrix, automated at scale
- **imatrix critical for IQ4_NL and below**: Without it, significant quality loss
- **Unsloth's imatrix is downloadable**: `imatrix_unsloth.gguf_file` from their HF repos
- **llama-quantize supports mixed precision**: `--tensor-type "pattern=type"` with regex matching

## What to do next (in priority order)

### 1. Build dev engine and test Codex CLI (IMMEDIATE)
The dev fork has PR #19720 (Responses API) merged but NOT rebuilt yet.
```bash
cmake --build ~/Documents/work/ai/local-llm/engine/dev/llama.cpp/build --config Release -j$(nproc)
llm switch interactive --dev
# Test Responses API
curl http://localhost:8100/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"default","input":"Say hello","max_output_tokens":50}'
# If that works, test Codex CLI
codex --model qwen3.5-9b --provider local
```

### 2. Set up Crush or Aider (works TODAY, no PR needed)
Both speak Chat Completions. Good alternatives if Codex/Responses API is flaky.
- **Crush** (Go/Bubbletea, 22.8K stars): `base_url` in JSON config
- **Aider** (Python, 43K stars): `aider --openai-api-base http://localhost:8100/v1 --openai-api-key none`

### 3. Test 35B on Pi (quality comparison)
Pi worked with 9B but had file-overwrite issues. Test with 35B to see if the larger model handles tool selection better.
```bash
llm switch workhorse
pi
```

### 4. Remaining from previous sessions
- Run per-project test suites on 9B/35B
- PR #21594: --reasoning-budget fix
- Reasoning ON for 35B (research thinking token budgets)
- Small model use cases (0.8B/2B/4B)
- Wiki updates

## Key commands (updated)

```bash
# All tiers now on port 8100
llm switch interactive              # 9B stable, 41 t/s
llm switch interactive --dev        # 9B dev engine (Responses API)
llm switch workhorse                # 35B regex, 21 t/s
llm switch workhorse-full           # 35B cmoe, 15 t/s, 262K

# Harnesses
pi                                  # Pi with local model (alias)
colo                                # Codex CLI no-sandbox (alias)
codex --model qwen3.5-9b --provider local  # Codex with local model

# Dev fork management
cd ~/Documents/work/ai/local-llm/engine/dev/llama.cpp
git log --oneline -5                # see what PRs are applied
# Current: PR #19720 (Responses API) on branch dev-prs
```
