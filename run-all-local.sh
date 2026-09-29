#!/usr/bin/env bash
# run-all-local.sh — Local Gemma Avatar master launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Starting Local Gemma Avatar Stack"
echo "============================================================"

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:31b}"
export LOCAL_LLM_REQUEST_TIMEOUT_S="${LOCAL_LLM_REQUEST_TIMEOUT_S:-300}"

# Make scripts executable
chmod +x start-ollama.sh start-speech-to-speech.sh start-avatar-frontend.sh test-tools.sh

echo "1/3 Starting Ollama LLM Service..."
./start-ollama.sh &
OLLAMA_PID=$!

echo "Waiting for Ollama to initialize..."
sleep 3

echo "2/3 Starting Speech-to-Speech Realtime Pipeline..."
./start-speech-to-speech.sh &
S2S_PID=$!

echo "Waiting for models to initialize..."
sleep 5

echo "3/3 Starting Gemma Avatar Frontend..."
echo "Open browser at: http://localhost:3000"
./start-avatar-frontend.sh

trap "kill $OLLAMA_PID $S2S_PID 2>/dev/null || true" EXIT
