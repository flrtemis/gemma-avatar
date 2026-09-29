#!/usr/bin/env bash
# test-tools.sh — Verify Ollama tool calling for Gemma Avatar gestures
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_URL="${1:-http://${OLLAMA_HOST}/v1/chat/completions}"
MODEL="${2:-${OLLAMA_MODEL:-gemma4:31b}}"

echo "Testing function calling on model ${MODEL} via ${OLLAMA_URL}..."

curl -s "${OLLAMA_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "stream": false,
    "messages": [
      {
        "role": "system",
        "content": "Use the supplied avatar tools when appropriate."
      },
      {
        "role": "user",
        "content": "Greet me enthusiastically and give me a thumbs up."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "make_hand_gesture",
          "description": "Make a hand gesture with the avatar.",
          "parameters": {
            "type": "object",
            "properties": {
              "gesture": {
                "type": "string",
                "enum": ["handup","index","ok","thumbup","thumbdown","side","shrug"]
              }
            },
            "required": ["gesture"]
          }
        }
      }
    ]
  }' | grep -q "tool_calls" && echo "SUCCESS: Ollama generated tool call!" || echo "RESPONSE RECEIVED (check tool support for model ${MODEL})."
