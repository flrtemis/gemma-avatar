#!/usr/bin/env bash
# start-speech-to-speech.sh — Launch local Hugging Face Speech-to-Speech server bridging Ollama to Realtime WS
#
# TTS engines:
#   TTS_ENGINE=qwen3 (default)  → Qwen3-TTS CustomVoice (bundled upstream handler)
#   TTS_ENGINE=xtts             → your own Coqui XTTS-v2 fine-tune (see TTS-XTTS.md)
#
# XTTS example:
#   TTS_ENGINE=xtts XTTS_VOICE_DIR="$HOME/voices/my-voice" ./start-speech-to-speech.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Force offline mode for pre-warmed HF caches
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

VENV_PATH="${VENV_PATH:-$HOME/venvs/gemma-avatar-s2s}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
LOCAL_LLM_REQUEST_TIMEOUT_S="${LOCAL_LLM_REQUEST_TIMEOUT_S:-300}"
TTS_ENGINE="${TTS_ENGINE:-${GEMMA_TTS_ENGINE:-qwen3}}"

# ── XTTS (Coqui) settings ─────────────────────────────────────────────
XTTS_VOICE_DIR="${XTTS_VOICE_DIR:-}"
XTTS_LANGUAGE="${XTTS_LANGUAGE:-en}"
XTTS_DEVICE="${XTTS_DEVICE:-auto}"
XTTS_STREAM="${XTTS_STREAM:-1}"
XTTS_STREAM_CHUNK_SIZE="${XTTS_STREAM_CHUNK_SIZE:-20}"
XTTS_WARMUP="${XTTS_WARMUP:-1}"
export XTTS_VOICE_DIR XTTS_LANGUAGE XTTS_DEVICE XTTS_STREAM XTTS_STREAM_CHUNK_SIZE XTTS_WARMUP

# ── Qwen3-TTS settings (used when TTS_ENGINE=qwen3) ───────────────────
QWEN3_GGUF_DIR="${QWEN3_GGUF_DIR:-$SCRIPT_DIR/models/qwen3-tts-gguf}"
QWEN3_GGUF_TALKER_PATH="${QWEN3_GGUF_TALKER_PATH:-$QWEN3_GGUF_DIR/qwen-talker-1.7b-customvoice-BF16.gguf}"
QWEN3_GGUF_CODEC_PATH="${QWEN3_GGUF_CODEC_PATH:-$QWEN3_GGUF_DIR/qwen-tokenizer-12hz-BF16.gguf}"
QWEN3_TTS_BACKEND="${QWEN3_TTS_BACKEND:-torch}"
QWEN3_TTS_MODEL_NAME="${QWEN3_TTS_MODEL_NAME:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
QWEN3_TTS_CACHE_ROOT="${QWEN3_TTS_CACHE_ROOT:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots}"

normalize_engine() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    xtts|xtts-v2|xtts_v2|coqui-xtts|coqui_xtts) printf 'xtts\n' ;;
    qwen3|qwen) printf 'qwen3\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

pick_model() {
  if [ -n "${1:-}" ]; then
    printf '%s\n' "$1"
    return
  fi
  if [ -n "${OLLAMA_MODEL:-}" ]; then
    printf '%s\n' "$OLLAMA_MODEL"
    return
  fi
  if ollama list 2>/dev/null | grep -q '^gemma4:31b'; then
    printf '%s\n' 'gemma4:31b'
    return
  fi
  if ollama list 2>/dev/null | grep -q '^gemma4:26b'; then
    printf '%s\n' 'gemma4:26b'
    return
  fi
  if ollama list 2>/dev/null | grep -q '^gemma4:12b'; then
    printf '%s\n' 'gemma4:12b'
    return
  fi
  if ollama list 2>/dev/null | grep -q '^gemma4:latest'; then
    printf '%s\n' 'gemma4:latest'
    return
  fi
  printf '%s\n' 'gemma4:31b'
}

if [ -f "$VENV_PATH/bin/activate" ]; then
  echo "Activating virtualenv at $VENV_PATH..."
  source "$VENV_PATH/bin/activate"
else
  echo "Warning: Virtualenv at $VENV_PATH not found. Using system Python."
fi

# Backwards compatible arguments: `start-speech-to-speech.sh [engine] [model]`.
# A first argument that is not a known engine is treated as the Ollama model name,
# so existing `./start-speech-to-speech.sh gemma4:26b` invocations keep working.
if [ $# -gt 0 ]; then
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    xtts|xtts-v2|xtts_v2|coqui-xtts|coqui_xtts|qwen3|qwen) TTS_ENGINE="$1"; shift ;;
  esac
fi
TTS_ENGINE="$(normalize_engine "$TTS_ENGINE")"
export GEMMA_TTS_ENGINE="$TTS_ENGINE"

MODEL_NAME="$(pick_model "${1:-}")"
OLLAMA_URL="${OLLAMA_URL:-http://${OLLAMA_HOST}/v1}"
export OLLAMA_MODEL="$MODEL_NAME"
export LOCAL_LLM_REQUEST_TIMEOUT_S

if [ "$TTS_ENGINE" = "xtts" ]; then
  if ! python -c "import TTS" >/dev/null 2>&1; then
    echo "ERROR: Coqui TTS is not installed in this Python environment."
    echo "       Install it with: pip install -r requirements-xtts.txt"
    exit 1
  fi
  if [ -z "${XTTS_VOICE_DIR:-}" ] && [ -z "${XTTS_MODEL_PATH:-}" ]; then
    echo "ERROR: TTS_ENGINE=xtts needs a voice folder."
    echo "       Set XTTS_VOICE_DIR to the folder holding model.pth/config.json/vocab.json, e.g."
    echo "         XTTS_VOICE_DIR=\"\$HOME/voices/my-voice\" ./start-speech-to-speech.sh xtts"
    exit 1
  fi
  if [ -f "$SCRIPT_DIR/check_xtts_voice.py" ]; then
    python "$SCRIPT_DIR/check_xtts_voice.py" ${XTTS_VOICE_DIR:+"$XTTS_VOICE_DIR"} --quiet || \
      echo "Warning: XTTS voice folder check reported problems (see above); starting anyway."
  fi
  echo "TTS engine: XTTS (Coqui) | voice dir: ${XTTS_VOICE_DIR:-<from XTTS_MODEL_PATH>} | language: $XTTS_LANGUAGE | device: $XTTS_DEVICE"
else
  if [ -f "$QWEN3_GGUF_TALKER_PATH" ] && [ -f "$QWEN3_GGUF_CODEC_PATH" ]; then
    export QWEN3_GGUF_TALKER_PATH
    export QWEN3_GGUF_CODEC_PATH
    echo "Using local GGUF TTS assets from $QWEN3_GGUF_DIR"
  elif [ "$QWEN3_TTS_BACKEND" = "ggml" ]; then
    echo "Local GGUF TTS assets not found; ggml backend will require either cached Hugging Face GGUF files or a one-time run of download_tts_assets.py."
  fi

  if [ "$QWEN3_TTS_BACKEND" = "torch" ] && [ -d "$QWEN3_TTS_CACHE_ROOT" ]; then
    QWEN3_TTS_LOCAL_SNAPSHOT="$(find "$QWEN3_TTS_CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -n "$QWEN3_TTS_LOCAL_SNAPSHOT" ]; then
      QWEN3_TTS_MODEL_NAME="$QWEN3_TTS_LOCAL_SNAPSHOT"
      echo "Using cached local Qwen3-TTS snapshot: $QWEN3_TTS_MODEL_NAME"
    fi
  fi
fi

echo "Starting speech-to-speech server on ws://127.0.0.1:8765/v1/realtime..."
echo "LLM Model: ${MODEL_NAME} | LLM Endpoint: ${OLLAMA_URL} | Timeout: ${LOCAL_LLM_REQUEST_TIMEOUT_S}s | TTS Backend: ${TTS_ENGINE}"

exec python ./local_s2s_launcher.py \
  --mode realtime \
  --thresh 0.6 \
  --stt parakeet-tdt \
  --llm_backend chat-completions \
  --tts "${TTS_ENGINE}" \
  --qwen3_tts_model_name "${QWEN3_TTS_MODEL_NAME}" \
  --qwen3_tts_speaker Sohee \
  --qwen3_tts_language auto \
  --qwen3_tts_backend "${QWEN3_TTS_BACKEND}" \
  --qwen3_tts_non_streaming_mode False \
  --model_name "${MODEL_NAME}" \
  --chat_size 30 \
  --responses_api_base_url "${OLLAMA_URL}" \
  --responses_api_api_key "ollama" \
  --responses_api_reasoning_effort none \
  --responses_api_stream \
  --enable_live_transcription
