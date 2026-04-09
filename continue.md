# Continue from here

Read decisions.md first — it has all benchmark data, architecture decisions, and commands.

## What happened

Over two sessions we built a complete local LLM benchmarking and serving infrastructure:

1. **Built llama.cpp** with CUDA 13.0 for RTX 3060 12GB
2. **Benchmarked 7 model sizes** (0.8B through 35B MoE) across multiple quants
3. **Found the two winning models**: 9B IQ4_NL (41 t/s, code gen) and 35B-A3B IQ4_NL (21 t/s, code review)
4. **Tested context scaling** from 35K to 241K tokens on real project codebases
5. **Discovered key optimizations**: --reasoning off (critical), Q8 KV cache with activation rotation, regex expert offloading for MoE, prompt caching (15x speedup on follow-ups)
6. **Built a test suite** with 21 per-project test files from real git history + 5 context scale tests using real tokenizer counts
7. **Compared heretic fine-tune** to base 9B: tie on quality, base more accurate on diagnosis

## What to work on next

Ask the user which of these to prioritize:

### More quality testing
- The heretic vs base comparison was only on one task (code review). More tasks needed: tool calling, Svelte 5 generation, error diagnosis, git diff replay.
- The per-project test suites (git replay, tool calling, cross-project SW fix) haven't been run on any model yet. These are in benchmarks/<project>/.
- Run the polecat 5-test suite on the new IQ4_NL quants (we only ran it on Q5_K_M and IQ2_M before).

### Harness setup
- Pi Coding Agent, OpenCode, or Gastown — none are set up yet.
- The `llm` script serves OpenAI-compatible API. Harness just points to localhost:8010 or 8100.
- Pi is proven with Qwen 3.5 but dev created a company. OpenCode needs investigation. Gastown is multi-agent (ambitious).

### llama.cpp PRs
- PR #20700: Qwen3.5 native MTP (multi-token prediction). Could 1.5-2x generation speed. Needs fork + cherry-pick + rebuild.
- PR #21594: --reasoning-budget fix. Currently the flag is ignored.

### Higher quant for 35B
- We tested IQ4_NL (17.8GB). The Q8_0 is 36.9GB — fits in 32GB RAM with -cmoe. Would improve quality significantly. Worth testing overnight.

### Reasoning ON for 35B winner
- We always tested with --reasoning off. The 35B-A3B might produce much better analysis WITH thinking tokens, especially on complex tasks (the 30s timing bug that no local model found). Need to research: what max_tokens budget, how to balance thinking vs output, quality impact. Internet research first.

### Use cases for small models (0.8B/2B/4B at 300+ t/s parallel)
- These failed at code review (5-10/25) but 300+ t/s aggregate is valuable. After harness setup, research what the community uses sub-4B models for: autocomplete, structured extraction, JSON formatting, tool routing, draft-then-refine pipelines, embeddings. There's a use case — we just haven't found it yet.

### Wiki / report updates
- The HTML wiki at research.visar.veron3.space has outdated numbers from before benchmarking. Needs update with real measured data.

## How to use subagents efficiently

This session bloated context with tool calls and polling. For the next session:
- Send research/web queries to background agents
- Don't poll for results — wait for task notifications
- For benchmarks: give the user commands to run manually instead of running them yourself (avoids zombie processes and timeout issues)
- Save ALL test results permanently via run-test.sh (writes to benchmarks/results/)
- Use Opus agents for quality evaluation — save their output to benchmarks/results/

## Key commands

```bash
# Switch tiers
llm switch interactive          # 9B, 41 t/s, 262K context
llm switch workhorse            # 35B regex, 21 t/s, 180K context
llm switch workhorse-full       # 35B -cmoe, 15 t/s, 262K context

# Run a test (saves results permanently)
~/Documents/llm/benchmarks/run-test.sh ~/Documents/llm/benchmarks/context-scale-35k.json
~/Documents/llm/benchmarks/run-test.sh ~/Documents/llm/benchmarks/km-explorer/context-dump.json --tag heretic-km

# Start server manually
~/Documents/llm/engine/llama.cpp/build/bin/llama-server \
  -m ~/Documents/llm/models/<model>.gguf \
  -ngl 99 --no-mmap --jinja --reasoning off \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 262144 --port 8200

# Check server stats
grep -a "prompt eval time\|       eval time" /path/to/server.log

# Tokenize a file (server must be running)
curl -sf http://localhost:8200/tokenize -H "Content-Type: application/json" -d @/tmp/tok-req.json | python3 -c "import sys,json; print(len(json.load(sys.stdin)['tokens']))"
```

## Models on disk

```
~/Documents/llm/models/
├── qwen3.5-0.8b-q4km.gguf         508MB
├── qwen3.5-0.8b-q8.gguf           775MB
├── qwen3.5-0.8b-iq2xxs.gguf       323MB
├── qwen3.5-2b-q4km.gguf           1.2GB
├── qwen3.5-2b-q8.gguf             1.9GB
├── qwen3.5-2b-iq2xxs.gguf         680MB
├── qwen3.5-4b-q4km.gguf           2.6GB
├── qwen3.5-4b-q2kxl.gguf          1.9GB
├── qwen3.5-4b-q8.gguf             4.5GB
├── qwen3.5-9b-q5km.gguf           6.2GB   (old default, replaced by IQ4_NL)
├── qwen3.5-9b-iq4nl.gguf          5.4GB   ← interactive tier winner
├── qwen3.5-9b-iq3xxs.gguf         4.0GB
├── qwen3.5-9b-heretic-i1-iq4nl.gguf 5.0GB ← heretic fine-tune (tie with base)
├── qwen3.5-27b-iq2m.gguf          9.5GB   (dead end, 35B MoE is better)
├── qwen3.5-35b-a3b-iq2m.gguf      10.6GB  (old, replaced by IQ4_NL)
├── qwen3.5-35b-a3b-iq3xxs.gguf    12.2GB
└── qwen3.5-35b-a3b-iq4nl.gguf     17.8GB  ← workhorse tier winner
```
