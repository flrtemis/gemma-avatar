---
title: Gemma Avatar
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
thumbnail: https://huggingface.co/spaces/victor/gemma-avatar/resolve/main/thumbnail.webp
short_description: Talk to Gemma 4 face to face, with a 3D lip-synced avatar
models:
  - google/gemma-4-31B-it
  - nvidia/parakeet-tdt-1.1b
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
---

# Gemma Avatar

Realtime voice chat with a 3D talking-head avatar. Same AI stack as the
[smolagents/hf-realtime-voice](https://huggingface.co/spaces/smolagents/hf-realtime-voice)
Space ([blog post](https://huggingface.co/blog/cerebras-gemma4-voice-ai)), but the
orb visualization is replaced by a [TalkingHead](https://github.com/met4citizen/TalkingHead)
3D avatar with real-time audio-driven lip-sync.

## The pipeline

```
you speak → silero-VAD → parakeet-tdt-1.1b (STT) → gemma-4-31B-it on Cerebras → Qwen3-TTS → avatar speaks
```

Transport is the OpenAI Realtime GA protocol over WebSocket against Hugging
Face's speech-to-speech backend: mic PCM16 @ 16 kHz goes up as
`input_audio_buffer.append`, TTS PCM16 @ 16 kHz comes back as
`response.output_audio.delta`, transcripts stream alongside.

## How the avatar works

- **Rendering / body language** — [TalkingHead](https://github.com/met4citizen/TalkingHead)
  (three.js). Blinking, breathing, idle sway, moods, hand gestures, and emoji
  expressions are its built-in animation system.
- **Lip-sync** — the backend sends raw PCM only (no word timings, no visemes),
  so the mouth is driven from the audio itself with
  [HeadAudio](https://github.com/met4citizen/HeadAudio): an AudioWorklet that
  classifies MFCC frames into Oculus visemes (~50 ms latency, fully in-browser).
  The s2s playback worklet is routed into TalkingHead's audio graph
  (`audioAnalyzerNode → audioSpeechGainNode → reverb → speakers`) and HeadAudio
  taps the speech gain node.
- **The model plays the avatar** — three function tools are declared to the
  backend: `set_mood`, `make_hand_gesture`, `make_facial_expression`. Gemma
  calls them mid-conversation (smiles when greeting, shrugs when unsure,
  thumbs-up when agreeing).
- **Choreography** — client statuses drive presence: the avatar makes eye
  contact when you start talking, gestures with its hands on new utterances,
  and barge-in clears the playback buffer so the mouth settles instantly.

## Run it

```bash
bun install

# Pick a backend (one of):
LOAD_BALANCER_URL=https://…            bun run dev   # a speech-to-speech load balancer
SESSION_PROXY_URL=https://…/api        bun run dev   # piggyback another deployment's /api (dev)
bun run dev                                          # direct mode: paste a ws:// URL in Settings
```

Open http://localhost:3000 and tap **Start talking**.

For the offline WSL stack in this repo, use the launchers instead:

```bash
./run-all-local.sh
```

The local launchers now default to `http://127.0.0.1:11434` for Ollama, prefer `gemma4:31b`, assume a slow local warmup by default with a 300 second OpenAI-compatible request timeout, and use the `torch` Qwen3-TTS backend by default so the app does not depend on separate GGUF downloads. Override with `OLLAMA_MODEL=...`, `LOCAL_LLM_REQUEST_TIMEOUT_S=...`, or `QWEN3_TTS_BACKEND=ggml` if needed.

`?fakemic=1` starts a session with a silent synthetic mic (useful for testing
the full loop without a microphone — trigger a reply from the console with
`getClient().requestResponse()`).

## Layout

```
index.ts                    Bun server: HTML import + /api/session proxy + static assets
index.html                  App shell (avatar hero, caption, subtitles, settings)
src/app.js                  Session wiring, tool executor, UI state
src/avatar.js               AvatarStage: TalkingHead + HeadAudio + choreography
src/s2s/s2s-ws-client.js    Realtime WS client (vendored from the Space; orb removed,
                            injectable output node + worklet base URL + shared-ctx close)
src/s2s/codec.js            PCM/base64 + transcript helpers (vendored, unchanged)
src/vendor/headaudio.min.mjs  HeadAudio node class (bundled)
public/worklets/            mic-capture + audio-playback AudioWorklets (vendored, unchanged)
public/vendor/              HeadAudio worklet processor + viseme model (runtime-loaded)
public/avatars/brunette.glb Default avatar (Ready Player Me; CC BY-NC 4.0 — non-commercial)
```

## Notes

- The avatar GLB must have a Mixamo-compatible rig plus ARKit and Oculus-viseme
  blend shapes. Ready Player Me avatars work with
  `?morphTargets=ARKit,Oculus%20Visemes` on the GLB URL.
- TalkingHead owns the AudioContext; the s2s client is handed `head.audioCtx`
  and never closes it.
- Everything animation-related runs on requestAnimationFrame — a backgrounded
  tab freezes the avatar (audio keeps playing).

## How to run it

- Open Command Prompt
- Type wsl
- Hit Enter
- Type cd ~
- Hit Enter

The last 2 commands, as well as their output, are shown below as proof it runs successfully:

```ansi
(base) l3ung@DESKTOP-D23P7Q4:~$ cd gemma-avatar
(base) l3ung@DESKTOP-D23P7Q4:~/gemma-avatar$ OLLAMA_MODEL=gemma4:31b PATH="$HOME/.bun/bin:$PATH" ./run-all-local.sh
============================================================
 Starting Local Gemma Avatar Stack
============================================================
1/3 Starting Ollama LLM Service...
Waiting for Ollama to initialize...
Starting Ollama service on http://127.0.0.1:11434...
Ollama is already running.
Checking model availability: gemma4:31b...
Ollama is ready for local Gemma inference.
2/3 Starting Speech-to-Speech Realtime Pipeline...
Waiting for models to initialize...
Activating virtualenv at /home/l3ung/venvs/gemma-avatar-s2s...
Using local GGUF TTS assets from /home/l3ung/gemma-avatar/models/qwen3-tts-gguf
Using cached local Qwen3-TTS snapshot: /home/l3ung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe
Starting speech-to-speech server on ws://127.0.0.1:8765/v1/realtime...
LLM Model: gemma4:31b | LLM Endpoint: http://127.0.0.1:11434/v1 | Timeout: 300s | TTS Backend: torch
3/3 Starting Gemma Avatar Frontend...
Open browser at: http://localhost:3000
Starting Gemma Avatar frontend on http://localhost:3000...
$ bun --hot ./index.ts
gemma-avatar listening on http://localhost:3000/
session backend: none — direct mode (set LOAD_BALANCER_URL or SESSION_PROXY_URL)
DeepFilterNet not available for audio enhancement: No module named 'df'
[nltk_data] Downloading package averaged_perceptron_tagger_eng to
[nltk_data]     /home/l3ung/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger_eng is already up-to-
[nltk_data]       date!
Using cache found in /home/l3ung/.cache/torch/hub/snakers4_silero-vad_master
2026-09-29 15:00:35,731 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Loading Parakeet TDT model: nvidia/parakeet-tdt-0.6b-v3 on cuda
/home/l3ung/venvs/gemma-avatar-s2s/lib/python3.13/site-packages/torch/nn/modules/rnn.py:1164: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1479.)
  result = _VF.lstm(
2026-09-29 15:00:45,824 - speech_to_speech.STT.parakeet_tdt_handler - INFO - nano-parakeet model loaded successfully on cuda
2026-09-29 15:00:45,824 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Live transcription enabled for Parakeet TDT (nano_parakeet)
2026-09-29 15:00:45,824 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Warming up ParakeetTDTSTTHandler
2026-09-29 15:00:45,896 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Model warmed up and ready
2026-09-29 15:00:45,927 - speech_to_speech.LLM.chat_completions_language_model - INFO - Warming up ChatCompletionsApiModelHandler
2026-09-29 15:01:37,540 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:01:37,711 - speech_to_speech.LLM.chat_completions_language_model - INFO - ChatCompletionsApiModelHandler:  warmed up! time: 51.782 s
2026-09-29 15:01:37,842 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Loading Qwen3-TTS model: /home/l3ung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe via faster-qwen3-tts (torch backend)
2026-09-29 15:01:38,088 - faster_qwen3_tts.model - INFO - Loading Qwen3-TTS model: /home/l3ung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe

/bin/sh: 1: sox: not found
2026-09-29 15:01:41,808 - sox - WARNING - SoX could not be found!

    If you do not have SoX, proceed here:
     - - - http://sox.sourceforge.net/ - - -

    If you do (or think that you should) have SoX, double-check your
    path variables.

`torch_dtype` is deprecated! Use `dtype` instead!
2026-09-29 15:01:42,296 - qwen_tts.core.models.configuration_qwen3_tts - INFO - speaker_encoder_config is None. Initializing talker model with default values
2026-09-29 15:01:42,298 - qwen_tts.core.models.configuration_qwen3_tts - INFO - talker_config is None. Initializing talker model with default values
2026-09-29 15:01:42,298 - qwen_tts.core.models.configuration_qwen3_tts - INFO - speaker_encoder_config is None. Initializing talker model with default values
2026-09-29 15:01:42,298 - qwen_tts.core.models.configuration_qwen3_tts - INFO - code_predictor_config is None. Initializing code_predictor model with default values
2026-09-29 15:01:42,299 - qwen_tts.core.models.configuration_qwen3_tts - INFO - code_predictor_config is None. Initializing code_predictor model with default values
2026-09-29 15:01:48,994 - qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 - INFO - encoder_config is None. Initializing encoder with default values
2026-09-29 15:01:48,994 - qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 - INFO - decoder_config is None. Initializing decoder with default values
2026-09-29 15:01:52,962 - faster_qwen3_tts.model - INFO - Building CUDA graphs...
2026-09-29 15:01:52,972 - faster_qwen3_tts.model - INFO - CUDA graphs initialized (will capture on first run)
2026-09-29 15:01:52,972 - faster_qwen3_tts.model - WARNING - Could not infer sample rate from base model; defaulting to 24000 Hz.
2026-09-29 15:01:52,972 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS model loaded
2026-09-29 15:01:52,972 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Using Qwen3-TTS streaming chunk size 8 (~640ms audio per chunk) on faster_qwen3_tts
2026-09-29 15:01:52,972 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Warming up Qwen3TTSHandler
2026-09-29 15:01:52,972 - faster_qwen3_tts.model - INFO - Warming up CUDA graphs...
Warming up predictor (3 runs)...
Capturing CUDA graph for predictor...
CUDA graph captured!
Warming up talker graph (3 runs)...
Capturing CUDA graph for talker decode...
Talker CUDA graph captured!
2026-09-29 15:01:55,294 - faster_qwen3_tts.model - INFO - CUDA graphs captured and ready
2026-09-29 15:02:01,117 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 5.82s (custom_voice)
2026-09-29 15:02:02,728 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 3.88s audio in 7.43s (RTF: 0.52, custom_voice)
2026-09-29 15:02:02,728 - speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3TTSHandler warmed up
2026-09-29 15:02:02,750 - speech_to_speech.api.openai_realtime.server - INFO - OpenAI Realtime API starting on ws://0.0.0.0:8765/v1/realtime (pool size 1)
INFO:     Started server process [599]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8765 (Press CTRL+C to quit)
Bundled page in 119ms: index.html
INFO:     127.0.0.1:32990 - "WebSocket /v1/realtime" [accepted]
INFO:     connection open
2026-09-29 15:02:11,253 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Client connected to pipeline 0 (session session_53046464f26548a1b04120c055ba0afa)
2026-09-29 15:02:11,265 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.session - INFO - Session configuration updated
2026-09-29 15:02:32,394 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Client session_53046464f26548a1b04120c055ba0afa disconnected from pipeline 0
2026-09-29 15:02:32,446 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Session session_53046464f26548a1b04120c055ba0afa unregistered — cumulative: input_tokens=0, output_tokens=0, audio=0.00s
2026-09-29 15:02:32,446 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0 released (session session_53046464f26548a1b04120c055ba0afa ended)
INFO:     127.0.0.1:44282 - "WebSocket /v1/realtime" [accepted]
INFO:     connection open
2026-09-29 15:02:36,352 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Client connected to pipeline 0 (session session_5f2901594cb448efbc41d16d53af2944)
2026-09-29 15:02:36,370 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.session - INFO - Session configuration updated
2026-09-29 15:04:23,427 - [pipeline 0] local_s2s_launcher - INFO - upload received: name=Piece of mAInd.png mime=image/png bytes=1851032
2026-09-29 15:04:33,290 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:04:33,497 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 9.999 s
2026-09-29 15:04:33,498 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"mood":"love"}', call_id='call_0f736a4496fb4bff96a78de80a1d630d', name='set_mood', type='function_call', id='fc_08a7d42cf0aa496aaabdaf38583b8d6c', namespace=None, status='completed')]
2026-09-29 15:04:33,498 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['set_mood']
2026-09-29 15:04:33,500 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=954, output=15
2026-09-29 15:04:33,501 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=954, output_tokens=15, audio=0.00s | cumulative: input_tokens=954, output_tokens=15, audio=0.00s
2026-09-29 15:04:33,501 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:04:33,502 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_0f736a4496fb4bff96a78de80a1d630d
2026-09-29 15:04:35,648 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:04:38,034 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 4.532 s
2026-09-29 15:04:38,034 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
2026-09-29 15:04:38,051 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=724, output=17
ASSISTANT: Hello there! It's so lovely to see you.
2026-09-29 15:04:39,228 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.18s (custom_voice)
2026-09-29 15:04:42,232 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 3.40s audio in 4.18s (RTF: 0.81, custom_voice)
2026-09-29 15:04:42,255 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=724, output_tokens=17, audio=0.00s | cumulative: input_tokens=1678, output_tokens=32, audio=0.00s
2026-09-29 15:04:42,255 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:05:50,898 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Client session_5f2901594cb448efbc41d16d53af2944 disconnected from pipeline 0
2026-09-29 15:05:50,949 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Session session_5f2901594cb448efbc41d16d53af2944 unregistered — cumulative: input_tokens=1678, output_tokens=32, audio=0.00s
2026-09-29 15:05:50,949 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0 released (session session_5f2901594cb448efbc41d16d53af2944 ended)
INFO:     127.0.0.1:35724 - "WebSocket /v1/realtime" [accepted]
INFO:     connection open
2026-09-29 15:05:55,047 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Client connected to pipeline 0 (session session_2b46dc26b38e4fb892a2be4237352ee5)
2026-09-29 15:05:55,048 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.session - INFO - Session configuration updated
2026-09-29 15:05:56,636 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=384ms, min=384ms, segment=384ms, turn=turn_1 rev=0)
2026-09-29 15:05:56,956 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1204ms, active=608ms, turn=turn_1 rev=0)
2026-09-29 15:05:56,956 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_1 rev=0 audio=1.204s age=0.000s
2026-09-29 15:05:57,048 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_1 rev=0 total=0.092s lock_scope=0.092s inference=0.092s chars=16
USER: Can you hear me?
Language: en
2026-09-29 15:05:57,049 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.092 s
2026-09-29 15:05:57,049 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): Can you hear me?
2026-09-29 15:06:02,226 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:06:02,463 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 5.406 s
2026-09-29 15:06:02,463 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"mood":"happy"}', call_id='call_cc74dfa56e3a45d18e8ed77798b4a104', name='set_mood', type='function_call', id='fc_0537411c926d486b8ae95569517dea10', namespace=None, status='completed')]
2026-09-29 15:06:02,464 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['set_mood']
2026-09-29 15:06:02,467 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=701, output=15
2026-09-29 15:06:02,467 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=701, output_tokens=15, audio=1.20s | cumulative: input_tokens=2379, output_tokens=47, audio=1.20s
2026-09-29 15:06:02,468 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:06:02,470 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_cc74dfa56e3a45d18e8ed77798b4a104
2026-09-29 15:06:04,642 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:06:06,679 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 4.209 s
2026-09-29 15:06:06,679 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: Yes, I can hear you loud and clear!
2026-09-29 15:06:06,699 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=729, output=15
2026-09-29 15:06:07,960 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.28s (custom_voice)
2026-09-29 15:06:07,960 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 11.004s (turn=turn_1 rev=0)
2026-09-29 15:06:11,461 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 3.32s audio in 4.78s (RTF: 0.69, custom_voice)
2026-09-29 15:06:11,478 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=729, output_tokens=15, audio=0.00s | cumulative: input_tokens=3108, output_tokens=62, audio=1.20s
2026-09-29 15:06:11,478 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:06:16,556 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=384ms, min=384ms, segment=384ms, turn=turn_2 rev=0)
2026-09-29 15:06:17,717 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=2036ms, active=1440ms, turn=turn_2 rev=0)
2026-09-29 15:06:17,718 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_2 rev=0 audio=2.036s age=0.000s
2026-09-29 15:06:17,938 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_2 rev=0 total=0.220s lock_scope=0.220s inference=0.220s chars=33
USER: I was just checking to make sure.
Language: en
2026-09-29 15:06:17,939 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.221 s
2026-09-29 15:06:17,939 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): I was just checking to make sure.
2026-09-29 15:06:23,109 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:06:23,310 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 5.362 s
2026-09-29 15:06:23,310 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"emoji":"😊"}', call_id='call_00b9582848d047e3a86e4ffff8147901', name='make_facial_expression', type='function_call', id='fc_c8a97a9a13264d6d8d0dffdaf89050b2', namespace=None, status='completed')]
2026-09-29 15:06:23,310 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['make_facial_expression']
2026-09-29 15:06:23,312 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=761, output=17
2026-09-29 15:06:23,312 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=761, output_tokens=17, audio=2.04s | cumulative: input_tokens=3869, output_tokens=79, audio=3.24s
2026-09-29 15:06:23,312 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:06:23,314 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_00b9582848d047e3a86e4ffff8147901
2026-09-29 15:06:24,682 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:06:25,679 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 2.365 s
2026-09-29 15:06:25,679 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: No problem at all!
2026-09-29 15:06:25,691 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=792, output=6
2026-09-29 15:06:26,710 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.03s (custom_voice)
2026-09-29 15:06:26,711 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 8.994s (turn=turn_2 rev=0)
2026-09-29 15:06:27,132 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 1.88s audio in 1.45s (RTF: 1.29, custom_voice)
2026-09-29 15:06:27,183 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=792, output_tokens=6, audio=0.00s | cumulative: input_tokens=4661, output_tokens=85, audio=3.24s
2026-09-29 15:06:27,183 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:06:36,477 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=384ms, min=384ms, segment=384ms, turn=turn_3 rev=0)
2026-09-29 15:06:37,475 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1876ms, active=1280ms, turn=turn_3 rev=0)
2026-09-29 15:06:37,541 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet progressive STT timing turn=turn_3 rev=0 audio=0.884s age=0.001s lock_scope=1.063s inference=1.063s chars=15
2026-09-29 15:06:37,541 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - ParakeetSTT-Progressive: compute lock released after holding 1.06s
2026-09-29 15:06:37,541 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_3 rev=0 audio=1.876s age=0.067s
2026-09-29 15:06:37,598 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_3 rev=0 total=0.057s lock_scope=0.057s inference=0.057s chars=23
USER: Can I show you a photo?
Language: en
2026-09-29 15:06:37,599 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.058 s
2026-09-29 15:06:37,599 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): Can I show you a photo?
2026-09-29 15:06:42,805 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:06:42,995 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 5.385 s
2026-09-29 15:06:42,995 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"gesture":"ok"}', call_id='call_c8f2344ebbf54d4db7a61b676ac09525', name='make_hand_gesture', type='function_call', id='fc_1508da38e9dd4764b4252fd8a28d0aff', namespace=None, status='completed')]
2026-09-29 15:06:42,996 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['make_hand_gesture']
2026-09-29 15:06:43,002 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=818, output=17
2026-09-29 15:06:43,002 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=818, output_tokens=17, audio=1.88s | cumulative: input_tokens=5479, output_tokens=102, audio=5.12s
2026-09-29 15:06:43,003 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:06:43,004 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_c8f2344ebbf54d4db7a61b676ac09525
2026-09-29 15:06:44,457 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:06:46,910 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 3.905 s
2026-09-29 15:06:46,913 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: I'd love to see it, go ahead!
2026-09-29 15:06:46,934 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=849, output=12
2026-09-29 15:06:47,946 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.03s (custom_voice)
2026-09-29 15:06:47,947 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 10.473s (turn=turn_3 rev=0)
2026-09-29 15:06:48,585 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 2.51s audio in 1.67s (RTF: 1.50, custom_voice)
2026-09-29 15:06:48,630 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=849, output_tokens=12, audio=0.00s | cumulative: input_tokens=6328, output_tokens=114, audio=5.12s
2026-09-29 15:06:48,631 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:06:53,076 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=384ms, min=384ms, segment=384ms, turn=turn_4 rev=0)
2026-09-29 15:06:53,690 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet progressive STT timing turn=turn_4 rev=0 audio=0.884s age=0.000s lock_scope=0.614s inference=0.614s chars=0
2026-09-29 15:06:53,691 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - ParakeetSTT-Progressive: compute lock released after holding 0.61s
2026-09-29 15:06:54,035 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1844ms, active=1248ms, turn=turn_4 rev=0)
2026-09-29 15:06:54,036 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_4 rev=0 audio=1.844s age=0.000s
2026-09-29 15:06:54,062 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_4 rev=0 total=0.026s lock_scope=0.026s inference=0.026s chars=0
2026-09-29 15:06:54,062 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.026 s
2026-09-29 15:06:54,397 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: pending reopen candidate for speculative turn turn_4 revision 1
2026-09-29 15:06:54,556 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: reopened speculative turn turn_4 revision 1
2026-09-29 15:06:54,556 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=192ms, min=192ms, segment=192ms, turn=turn_4 rev=1)
2026-09-29 15:06:55,237 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1216ms, active=768ms, turn=turn_4 rev=1)
2026-09-29 15:06:55,237 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_4 rev=1 audio=3.060s age=0.000s
2026-09-29 15:06:55,305 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_4 rev=1 total=0.068s lock_scope=0.068s inference=0.068s chars=38
USER: When I show you photos, please tell me
Language: en
2026-09-29 15:06:55,306 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.069 s
2026-09-29 15:06:55,306 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): When I show you photos, please tell me
2026-09-29 15:06:56,155 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: pending reopen candidate for speculative turn turn_4 revision 2
2026-09-29 15:06:56,316 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: reopened speculative turn turn_4 revision 2
2026-09-29 15:06:56,316 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=192ms, min=192ms, segment=192ms, turn=turn_4 rev=2)
2026-09-29 15:06:56,319 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: speech during pending response: cancelled, queue flushed
2026-09-29 15:06:57,435 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1812ms, active=1216ms, turn=turn_4 rev=2)
2026-09-29 15:06:57,916 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: pending reopen candidate for speculative turn turn_4 revision 3
2026-09-29 15:06:58,074 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: reopened speculative turn turn_4 revision 3
2026-09-29 15:06:58,074 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=192ms, min=192ms, segment=192ms, turn=turn_4 rev=3)
2026-09-29 15:06:59,355 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1952ms, active=1408ms, turn=turn_4 rev=3)
2026-09-29 15:06:59,405 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet progressive STT timing turn=turn_4 rev=2 audio=4.808s age=0.000s lock_scope=2.050s inference=2.050s chars=61
2026-09-29 15:06:59,405 - [pipeline 0] speech_to_speech.STT.base_stt_handler - INFO - ParakeetTDTSTTHandler: dropping stale STT output for turn=turn_4 rev=2 age=0.000s
2026-09-29 15:06:59,405 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - ParakeetSTT-Progressive: compute lock released after holding 2.05s
2026-09-29 15:06:59,405 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_4 rev=3 audio=6.824s age=0.049s
2026-09-29 15:06:59,510 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_4 rev=3 total=0.105s lock_scope=0.105s inference=0.105s chars=94
USER: When I show you photos, please tell me a detailed description about what you see in the photo.
Language: en
2026-09-29 15:06:59,511 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.106 s
2026-09-29 15:06:59,511 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): When I show you photos, please tell me a detailed description about what you see in the photo.
2026-09-29 15:06:59,876 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: pending reopen candidate for speculative turn turn_4 revision 4
2026-09-29 15:07:00,036 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: reopened speculative turn turn_4 revision 4
2026-09-29 15:07:00,036 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=192ms, min=192ms, segment=192ms, turn=turn_4 rev=4)
2026-09-29 15:07:00,043 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: speech during pending response: cancelled, queue flushed
2026-09-29 15:07:00,315 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=928ms, active=352ms, turn=turn_4 rev=4)
2026-09-29 15:07:00,316 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_4 rev=4 audio=7.752s age=0.000s
2026-09-29 15:07:00,358 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_4 rev=4 total=0.043s lock_scope=0.043s inference=0.043s chars=100
USER: When I show you photos, please tell me a detailed description about what you see in the photo. Okay?
Language: en
2026-09-29 15:07:00,359 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.043 s
2026-09-29 15:07:00,359 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): When I show you photos, please tell me a detailed description about what you see in the photo. Okay?
2026-09-29 15:07:03,783 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:07:03,995 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - LLM generation cancelled (interruption)
2026-09-29 15:07:03,996 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Skipping stale LLM request for turn=turn_4 rev=3
2026-09-29 15:07:08,766 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:07:08,962 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 4.966 s
2026-09-29 15:07:08,962 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"gesture":"thumbup"}', call_id='call_c346827965954d1db2d176196bd6f2f5', name='make_hand_gesture', type='function_call', id='fc_d3c9b95789ad49fb979a52b7152b8a58', namespace=None, status='completed')]
2026-09-29 15:07:08,962 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['make_hand_gesture']
2026-09-29 15:07:08,966 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=896, output=18
2026-09-29 15:07:08,967 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=896, output_tokens=18, audio=7.75s | cumulative: input_tokens=7224, output_tokens=132, audio=12.87s
2026-09-29 15:07:08,967 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:07:08,971 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_c346827965954d1db2d176196bd6f2f5
2026-09-29 15:07:10,352 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:07:14,131 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 5.160 s
2026-09-29 15:07:14,132 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: You've got it, I'll give you a detailed description of everything I see.
2026-09-29 15:07:14,152 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=929, output=20
2026-09-29 15:07:15,387 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.25s (custom_voice)
2026-09-29 15:07:15,388 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 15.073s (turn=turn_4 rev=4)
2026-09-29 15:07:18,868 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 4.44s audio in 4.73s (RTF: 0.94, custom_voice)
2026-09-29 15:07:18,913 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=929, output_tokens=20, audio=0.00s | cumulative: input_tokens=8153, output_tokens=152, audio=12.87s
2026-09-29 15:07:18,913 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:08:01,934 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.session - INFO - Session configuration updated
2026-09-29 15:08:10,245 - [pipeline 0] local_s2s_launcher - INFO - upload received: name=Piece of mAInd.png mime=image/png bytes=1851032
2026-09-29 15:08:19,008 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:08:19,255 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 8.960 s
2026-09-29 15:08:19,256 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"mood":"neutral"}', call_id='call_4c269dc8c213403d933fdb7a0f4bddde', name='set_mood', type='function_call', id='fc_8e88bf3cc67c45e3aa9e47ffff1d67fb', namespace=None, status='completed')]
2026-09-29 15:08:19,256 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['set_mood']
2026-09-29 15:08:19,259 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=1246, output=15
2026-09-29 15:08:19,259 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=1246, output_tokens=15, audio=0.00s | cumulative: input_tokens=9399, output_tokens=167, audio=12.87s
2026-09-29 15:08:19,259 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:08:19,260 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_4c269dc8c213403d933fdb7a0f4bddde
2026-09-29 15:08:21,498 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:08:35,707 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 16.447 s
ASSISTANT: This is a high-angle, close-up shot of a piece of cake on a small white plate. It looks like a slice of layered cake with a creamy
white frosting and a generous layer of fruit jam or preserve in the middle. On top, there's a swirl of white cream and a single, bright red
raspberry.
2026-09-29 15:08:37,006 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.29s (custom_voice)
2026-09-29 15:08:37,007 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 96.692s (turn=turn_4 rev=4)
2026-09-29 15:08:56,160 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 20.28s audio in 20.45s (RTF: 0.99, custom_voice)
2026-09-29 15:09:00,023 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 24.311 s
2026-09-29 15:09:00,023 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: The cake is set against a soft, blurred background of a wooden surface, making the dessert the main focus.
2026-09-29 15:09:00,034 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=1016, output=94
2026-09-29 15:09:01,038 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 1.01s (custom_voice)
2026-09-29 15:09:01,039 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 120.724s (turn=turn_4 rev=4)
2026-09-29 15:09:04,714 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 5.71s audio in 4.69s (RTF: 1.22, custom_voice)
2026-09-29 15:09:04,764 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=1016, output_tokens=94, audio=0.00s | cumulative: input_tokens=10415, output_tokens=261, audio=12.87s
2026-09-29 15:09:04,764 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:09:13,557 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=384ms, min=384ms, segment=384ms, turn=turn_5 rev=0)
2026-09-29 15:09:14,557 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1876ms, active=1280ms, turn=turn_5 rev=0)
2026-09-29 15:09:14,558 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_5 rev=0 audio=1.876s age=0.000s
2026-09-29 15:09:14,626 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_5 rev=0 total=0.068s lock_scope=0.068s inference=0.068s chars=26
USER: That's not correct at all.
Language: en
2026-09-29 15:09:14,627 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.069 s
2026-09-29 15:09:14,627 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): That's not correct at all.
2026-09-29 15:09:20,320 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:09:20,527 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 5.900 s
2026-09-29 15:09:20,528 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"gesture":"shrug"}', call_id='call_eaf069a70d0746d4bb0357e37065c331', name='make_hand_gesture', type='function_call', id='fc_2cc5c5ee799e46bd92a041049f257f37', namespace=None, status='completed')]
2026-09-29 15:09:20,531 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['make_hand_gesture']
2026-09-29 15:09:20,536 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=1127, output=18
2026-09-29 15:09:20,536 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=1127, output_tokens=18, audio=1.88s | cumulative: input_tokens=11542, output_tokens=279, audio=14.74s
2026-09-29 15:09:20,537 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:09:20,538 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_eaf069a70d0746d4bb0357e37065c331
2026-09-29 15:09:21,940 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:09:27,227 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 6.688 s
2026-09-29 15:09:27,227 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: I'm sorry about that, I must have misread it. Could you tell me what's actually in the photo?
2026-09-29 15:09:27,239 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=1159, output=27
2026-09-29 15:09:28,186 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 0.96s (custom_voice)
2026-09-29 15:09:28,187 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 13.630s (turn=turn_5 rev=0)
2026-09-29 15:09:31,946 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 6.20s audio in 4.72s (RTF: 1.31, custom_voice)
2026-09-29 15:09:31,984 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=1159, output_tokens=27, audio=0.00s | cumulative: input_tokens=12701, output_tokens=306, audio=14.74s
2026-09-29 15:09:31,985 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:09:45,596 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=384ms, min=384ms, segment=384ms, turn=turn_6 rev=0)
2026-09-29 15:09:49,237 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=4500ms, active=3872ms, turn=turn_6 rev=0)
2026-09-29 15:09:49,237 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_6 rev=0 audio=4.500s age=0.000s
2026-09-29 15:09:49,297 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_6 rev=0 total=0.060s lock_scope=0.060s inference=0.060s chars=64
USER: I'm going to end our conversation by clicking the button labeled
Language: en
2026-09-29 15:09:49,298 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.060 s
2026-09-29 15:09:49,298 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): I'm going to end our conversation by clicking the button labeled
2026-09-29 15:09:49,435 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: pending reopen candidate for speculative turn turn_6 revision 1
2026-09-29 15:09:49,595 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: reopened speculative turn turn_6 revision 1
2026-09-29 15:09:49,595 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=192ms, min=192ms, segment=192ms, turn=turn_6 rev=1)
2026-09-29 15:09:49,599 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: speech during pending response: cancelled, queue flushed
2026-09-29 15:09:50,876 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=1632ms, active=1344ms, turn=turn_6 rev=1)
2026-09-29 15:09:50,876 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_6 rev=1 audio=6.132s age=0.000s
2026-09-29 15:09:50,919 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_6 rev=1 total=0.043s lock_scope=0.043s inference=0.043s chars=86
USER: I'm going to end our conversation by clicking the button labeled End Conversation Now.
Language: en
2026-09-29 15:09:50,921 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.045 s
2026-09-29 15:09:50,921 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): I'm going to end our conversation by clicking the button labeled End Conversation Now.
2026-09-29 15:09:51,195 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: pending reopen candidate for speculative turn turn_6 revision 2
2026-09-29 15:09:51,356 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - VAD: reopened speculative turn turn_6 revision 2
2026-09-29 15:09:51,356 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech started (confirmed, active=192ms, min=192ms, segment=192ms, turn=turn_6 rev=2)
2026-09-29 15:09:51,357 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: speech during pending response: cancelled, queue flushed
2026-09-29 15:09:51,675 - [pipeline 0] speech_to_speech.VAD.vad_handler - INFO - Speech soft-ended (segment=800ms, active=416ms, turn=turn_6 rev=2)
2026-09-29 15:09:53,332 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet progressive STT timing turn=turn_6 rev=2 audio=6.612s age=0.000s lock_scope=1.976s inference=1.976s chars=92
2026-09-29 15:09:53,332 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - ParakeetSTT-Progressive: compute lock released after holding 1.98s
2026-09-29 15:09:53,332 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT start turn=turn_6 rev=2 audio=6.932s age=1.657s
2026-09-29 15:09:53,373 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Parakeet final STT done turn=turn_6 rev=2 total=0.041s lock_scope=0.041s inference=0.040s chars=92
USER: I'm going to end our conversation by clicking the button labeled End Conversation Now, okay?
Language: en
2026-09-29 15:09:53,374 - [pipeline 0] speech_to_speech.baseHandler - INFO - ParakeetTDTSTTHandler: 0.042 s
2026-09-29 15:09:53,374 - [pipeline 0] speech_to_speech.STT.transcription_notifier - INFO - Transcription completed (language=en): I'm going to end our conversation by clicking the button labeled End Conversation Now, okay?
2026-09-29 15:09:57,596 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:09:57,804 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - LLM generation cancelled (interruption)
2026-09-29 15:09:57,804 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Skipping stale LLM request for turn=turn_6 rev=1
2026-09-29 15:10:02,273 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:10:02,501 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 4.696 s
2026-09-29 15:10:02,507 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: [ResponseFunctionToolCall(arguments='{"emoji":"👋"}', call_id='call_744d4752571d449b889fa2c6fc87afee', name='make_facial_expression', type='function_call', id='fc_bb9f0a8882cf4f3cafd683f84b84c191', namespace=None, status='completed')]
2026-09-29 15:10:02,508 - [pipeline 0] speech_to_speech.LLM.lm_output_processor - INFO - Sending to clients: text='', tools=['make_facial_expression']
2026-09-29 15:10:02,509 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=1218, output=17
2026-09-29 15:10:02,509 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=1218, output_tokens=17, audio=6.93s | cumulative: input_tokens=13919, output_tokens=323, audio=21.68s
2026-09-29 15:10:02,509 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:10:02,510 - [pipeline 0] speech_to_speech.LLM.chat - INFO - Re-injecting evicted function_call for call_id=call_744d4752571d449b889fa2c6fc87afee
2026-09-29 15:10:03,806 - [pipeline 0] httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-29 15:10:05,911 - [pipeline 0] speech_to_speech.baseHandler - INFO - ChatCompletionsApiModelHandler: 3.400 s
2026-09-29 15:10:05,911 - [pipeline 0] speech_to_speech.LLM.base_openai_compatible_language_model - INFO - Tools: []
ASSISTANT: Sounds good, it was nice talking with you!
2026-09-29 15:10:05,929 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Token usage (response): input=1249, output=11
2026-09-29 15:10:06,725 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS TTFA: 0.81s (custom_voice)
2026-09-29 15:10:06,725 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Last speech detected to first speech out: 15.050s (turn=turn_6 rev=2)
2026-09-29 15:10:09,271 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS generated 2.97s audio in 3.36s (RTF: 0.88, custom_voice)
2026-09-29 15:10:09,305 - [pipeline 0] speech_to_speech.api.openai_realtime.handlers.response - INFO - Response done (status=completed) — this response: input_tokens=1249, output_tokens=11, audio=0.00s | cumulative: input_tokens=15168, output_tokens=334, audio=21.68s
2026-09-29 15:10:09,306 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0: response complete, listening re-enabled
2026-09-29 15:10:11,804 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Client session_2b46dc26b38e4fb892a2be4237352ee5 disconnected from pipeline 0
[browser] [ws] socket closed: 1000 client closed
2026-09-29 15:10:11,854 - [pipeline 0] speech_to_speech.api.openai_realtime.service - INFO - Session session_2b46dc26b38e4fb892a2be4237352ee5 unregistered — cumulative: input_tokens=15168, output_tokens=334, audio=21.68s
2026-09-29 15:10:11,854 - [pipeline 0] speech_to_speech.api.openai_realtime.websocket_router - INFO - Pipeline 0 released (session session_2b46dc26b38e4fb892a2be4237352ee5 ended)
^C
Shutting down gracefully...
2026-09-29 15:10:28,131 - [pipeline 0] speech_to_speech.STT.parakeet_tdt_handler - INFO - Cleaning up ParakeetTDTSTTHandler

(base) l3ung@DESKTOP-D23P7Q4:~/gemma-avatar$ INFO:     Shutting down
2026-09-29 15:10:28,219 - [pipeline 0] speech_to_speech.TTS.qwen3_tts_handler - INFO - Qwen3-TTS handler cleaned up
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [599]
✓ Pipeline stopped successfully
^C
(base) l3ung@DESKTOP-D23P7Q4:~/gemma-avatar$
```
