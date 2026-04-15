#!/usr/bin/env bash
set -euo pipefail

# Auto-detect running server from llm tier ports
if [[ -n "${LLM_URL:-}" ]]; then
  URL="$LLM_URL"
else
  URL=""
  for port in 8100; do
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
      URL="http://localhost:$port/v1/chat/completions"
      break
    fi
  done
  if [[ -z "$URL" ]]; then
    echo "No LLM server found on ports 8100, 8010. Start one with: llm switch <tier>"
    exit 1
  fi
fi
RESULTS_DIR="${LLM_RESULTS:-$HOME/Documents/work/ai/local-llm/benchmarks/results}"
mkdir -p "$RESULTS_DIR"

if [ $# -lt 1 ]; then
  echo "Usage: run-test.sh <test.json> [--thinking] [--tag <name>]"
  echo "Results saved to $RESULTS_DIR/<timestamp>-<tag>.json"
  exit 1
fi

FILE="$1"
shift
THINKING=""
TAG=$(basename "$FILE" .json)

while [ $# -gt 0 ]; do
  case "$1" in
    --thinking) THINKING="yes"; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    *) shift ;;
  esac
done

TMPDIR=$(mktemp -d /tmp/llm-test-XXXXXX)
trap "rm -rf $TMPDIR" EXIT

if [ -z "$THINKING" ]; then
  python3 -c "
import json
with open('$FILE') as f:
    d = json.load(f)
if 'chat_template_kwargs' not in d:
    d['chat_template_kwargs'] = {}
d['chat_template_kwargs']['enable_thinking'] = False
with open('$TMPDIR/request.json', 'w') as f:
    json.dump(d, f)
"
else
  cp "$FILE" "$TMPDIR/request.json"
fi

START=$(date +%s%N)
curl -s "$URL" -H "Content-Type: application/json" -d @"$TMPDIR/request.json" > "$TMPDIR/response.json"
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTFILE="$RESULTS_DIR/${TIMESTAMP}-${TAG}.json"

python3 - "$TMPDIR/response.json" "$MS" "$OUTFILE" "$FILE" << 'PYEOF'
import json, sys

resp_path, ms, out_path, input_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(resp_path) as f:
    r = json.load(f)

u = r.get('usage', {})
m = r['choices'][0]['message']
tc = m.get('tool_calls')
c = m.get('content', '')
fr = r['choices'][0].get('finish_reason', '?')

pp = u.get('prompt_tokens', '?')
ct = u.get('completion_tokens', '?')

result = {
    'input_file': input_path,
    'wall_ms': int(ms),
    'prompt_tokens': pp,
    'completion_tokens': ct,
    'finish_reason': fr,
    'tool_calls': tc,
    'content': c,
    'raw_usage': u
}

with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f'Prompt: {pp} tokens | Completion: {ct} tokens | Finish: {fr} | Wall: {ms}ms')
print(f'Saved: {out_path}')
print()
if tc:
    print('Tool calls:')
    for t in tc:
        print(f'  {t["function"]["name"]}({t["function"]["arguments"]})')
if c:
    print('Response (first 500 chars):')
    print(c[:500])
PYEOF
