# Engine Layout

This directory hosts local inference engine infrastructure used by the parent
`local-llm` workspace.

## Structure

- `stable/llama.cpp/` — nested git repo tracking upstream `ggml-org/llama.cpp`
- `dev/llama.cpp/` — nested git repo for local PR experiments and cherry-picks
- `venv/` — local Python environment for conversion/build helpers; not versioned

## Ownership

The parent `local-llm` repo documents how the engines are used, but the two
`llama.cpp` directories are separate git repos with their own history/remotes.
Do not treat them as normal folders in the parent repo.

## Build Output

Generated build artifacts live in:

- `stable/llama.cpp/build/`
- `dev/llama.cpp/build/`

These are intentionally ignored by the parent repo.
