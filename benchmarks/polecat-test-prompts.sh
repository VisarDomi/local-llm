#!/usr/bin/env bash
# Polecat benchmark: 5 realistic agentic test prompts for small LLMs (0.8B-2B)
# Target: http://localhost:8200/v1/chat/completions
# Each prompt tests a different fast-agent capability

set -euo pipefail

URL="http://localhost:8200/v1/chat/completions"

run_test() {
    local test_num="$1"
    local json_body="$2"

    echo ""
    echo "--- Test $test_num ---"
    local start end elapsed_ms
    start=$(date +%s%N)
    local response
    response=$(curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$json_body")
    end=$(date +%s%N)
    elapsed_ms=$(( (end - start) / 1000000 ))

    local prompt_tokens completion_tokens content tool_calls finish
    prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // "N/A"')
    completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // "N/A"')
    content=$(echo "$response" | jq -r '.choices[0].message.content // empty')
    tool_calls=$(echo "$response" | jq -r '.choices[0].message.tool_calls // empty')
    finish=$(echo "$response" | jq -r '.choices[0].finish_reason // "N/A"')

    echo "Time: ${elapsed_ms}ms | Prompt: $prompt_tokens | Completion: $completion_tokens | Finish: $finish"

    if [ -n "$tool_calls" ] && [ "$tool_calls" != "null" ]; then
        echo "Tool calls:"
        echo "$tool_calls" | jq -c '.[]'
    fi

    if [ -n "$content" ] && [ "$content" != "null" ]; then
        echo "Response (first 500 chars):"
        echo "$content" | head -c 500
        echo ""
    fi
}

echo "Running 5 Polecat benchmark tests..."
echo "================================================="

#=============================================================================
# TEST 1: Tool call - model must emit a proper function call
#=============================================================================
BODY1=$(cat <<'ENDJSON'
{
  "model": "polecat",
  "max_tokens": 256,
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a coding assistant. You have access to tools. When you need to find information in the codebase, use the available tools. Always respond with a tool call when you need to search or read files."
    },
    {
      "role": "user",
      "content": "I need to find all files in the manga-reader project that import from fetchJson.ts. The project is at ~/Documents/work/manga/manga-reader/. Use the grep tool to search for this import pattern."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "grep_search",
        "description": "Search for a pattern in files using ripgrep",
        "parameters": {
          "type": "object",
          "properties": {
            "pattern": { "type": "string", "description": "Regex pattern to search for" },
            "path": { "type": "string", "description": "Directory to search in" },
            "glob": { "type": "string", "description": "File glob filter, e.g. *.ts" }
          },
          "required": ["pattern", "path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": { "type": "string", "description": "Absolute path to the file" }
          },
          "required": ["file_path"]
        }
      }
    }
  ]
}
ENDJSON
)
run_test 1 "$BODY1"

#=============================================================================
# TEST 2: Code review - real function from video-platform api.ts
#=============================================================================
BODY2=$(cat <<'ENDJSON'
{
  "model": "polecat",
  "max_tokens": 256,
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a senior TypeScript code reviewer. Be concise. List only real bugs or issues, not style preferences."
    },
    {
      "role": "user",
      "content": "Any bugs in this API module from a video platform SvelteKit app?\n\n```typescript\nimport { API } from '../constants.js';\nimport type { Video } from '../types.js';\n\nexport class ApiError extends Error {\n  constructor(public status: number, message: string) {\n    super(message);\n  }\n}\n\nexport async function fetchVideos(provider: string): Promise<Video[]> {\n  const url = `${API.VIDEOS}?provider=${encodeURIComponent(provider)}`;\n  const response = await fetch(url);\n  const items: Omit<Video, 'provider'>[] = await response.json();\n  return items.map((v) => ({ ...v, provider }));\n}\n\nexport async function fetchNewVideos(provider: string, after: string): Promise<Video[]> {\n  const url = `${API.VIDEOS}?provider=${encodeURIComponent(provider)}&after=${encodeURIComponent(after)}`;\n  const response = await fetch(url);\n  const items: Omit<Video, 'provider'>[] = await response.json();\n  return items.map((v) => ({ ...v, provider }));\n}\n\nexport async function sendEditRequest(filename: string, segments: string[], provider: string): Promise<void> {\n  const response = await fetch(API.EDIT, {\n    method: 'POST',\n    headers: { 'Content-Type': 'application/json' },\n    body: JSON.stringify({ filename, segments, provider })\n  });\n  if (!response.ok) throw new ApiError(response.status, `Edit failed: ${response.statusText}`);\n}\n```"
    }
  ]
}
ENDJSON
)
run_test 2 "$BODY2"

#=============================================================================
# TEST 3: Quick refactor - add max toast limit and per-toast cleanup
#=============================================================================
BODY3=$(cat <<'ENDJSON'
{
  "model": "polecat",
  "max_tokens": 256,
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a TypeScript refactoring assistant. Output only the refactored code, no explanations."
    },
    {
      "role": "user",
      "content": "Refactor this Svelte 5 ToastState class to add a max toast limit of 5 (dismiss oldest when exceeded) and make the timer cleanup automatic per-toast instead of only in destroy(). Keep the $state rune.\n\n```typescript\nexport class ToastState {\n    items = $state<{ id: number; message: string }[]>([]);\n    private nextId = 0;\n    private timerIds = new Set<ReturnType<typeof setTimeout>>();\n\n    show(message: string, duration = 2000) {\n        const id = this.nextId++;\n        this.items = [...this.items, { id, message }];\n        const timerId = setTimeout(() => {\n            this.timerIds.delete(timerId);\n            this.items = this.items.filter(t => t.id !== id);\n        }, duration);\n        this.timerIds.add(timerId);\n    }\n\n    destroy() {\n        for (const t of this.timerIds) clearTimeout(t);\n        this.timerIds.clear();\n        this.items = [];\n    }\n}\n```"
    }
  ]
}
ENDJSON
)
run_test 3 "$BODY3"

#=============================================================================
# TEST 4: Error diagnosis - Cloudflare proxy timeout
#=============================================================================
BODY4=$(cat <<'ENDJSON'
{
  "model": "polecat",
  "max_tokens": 256,
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a debugging assistant for a TypeScript/Svelte 5 monorepo. Diagnose errors concisely."
    },
    {
      "role": "user",
      "content": "My manga-reader app crashes with this error when loading chapter images. The app uses a proxy server that forwards requests through a headless browser for Cloudflare-protected sites. What is wrong and how do I fix it?\n\n```\nApiError: HTTP 503\n    at doFetch (fetchJson.ts:56:23)\n    at async proxyRequest (api.ts:47:16)\n    at async fetchChapterImages (api.ts:177:18)\n\nAdditional context from server logs:\n[manga-reader] POST /api/proxy {url: \"https://example-manga.com/api/chapters/12345/images\", cloudflareProtected: true}\n[manga-reader] Cloudflare challenge detected, solving...\n[manga-reader] Error: Navigation timeout of 30000ms exceeded\n[manga-reader]   at CloakBrowser.solve (/packages/server/src/cloudflare.ts:89:11)\n[manga-reader] Returning 503 with X-Cloudflare-Solving: true\n```\n\nThe client-side code has Cloudflare retry logic (6 retries x 5s = 30s wait) but it still fails."
    }
  ]
}
ENDJSON
)
run_test 4 "$BODY4"

#=============================================================================
# TEST 5: Generation - Svelte 5 rune-based debounced search utility
#=============================================================================
BODY5=$(cat <<'ENDJSON'
{
  "model": "polecat",
  "max_tokens": 256,
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a Svelte 5 expert. Use runes ($state, $derived, $effect). Output only the code in a single TypeScript file, no markdown fences, no explanations."
    },
    {
      "role": "user",
      "content": "Create a Svelte 5 reactive debounced search state class for a manga/gallery reader app. Requirements:\n- Class named DebouncedSearch in a .svelte.ts file\n- Has a `query` field using $state (the raw input)\n- Has a `debouncedQuery` field using $state that updates 300ms after `query` stops changing\n- Has an `isTyping` $derived field that is true when query !== debouncedQuery\n- Has a `clear()` method that resets both immediately\n- Constructor takes an optional initial query string\n- Must clean up the timeout (provide a destroy method)"
    }
  ]
}
ENDJSON
)
run_test 5 "$BODY5"

echo ""
echo "================================================="
echo "Benchmark complete."
