# README.md - Local Offline Avatar
## 🎙️ Local Offline Avatar: The Pure-Local Voice AI Space

"AI should be accessible to everyone, everywhere, without the gatekeeping of cloud dependencies or the recurring costs of API keys. This project is a testament to the power of the local open-source stack."

This repository contains a fully patched, frozen, and ready-to-run local implementation of the Gemma Avatar system. It is designed to run entirely within a WSL2 or Linux environment, providing real-time voice interaction with a 3D avatar without any external API calls.
------------------------------
## 🏗️ Architecture Overview
The system bridges a Windows-based browser frontend with a high-performance speech-to-speech backend running in WSL2.

Windows Browser (Chrome/Edge) @ http://localhost:3000
  │
  ├─ Microphone Audio (PCM16 16kHz base64)
  │     ▼
  ├─ Local WebSocket @ ws://localhost:8765/v1/realtime
  │     │ (Hugging Face speech-to-speech server in WSL2)
  │     ├─ Silero VAD (Voice Activity Detection)
  │     ├─ Parakeet-TDT 1.1B STT
  │     ├─ Ollama Gemma 4 (via local OpenAI-compatible Chat API)
  │     └─ Qwen3-TTS 1.7B (CustomVoice - default speaker: Sohee)
  │     ▼
  └─ Returned PCM16 16kHz audio stream + tool calls
        │
        ├─ Audio Worklet -> HeadAudio Viseme Classifier (model-en-mixed.bin)
        ├─ TalkingHead 3D Avatar Stage (brunette.glb)
        └─ Facial Expressions, Lip-Sync, and Gestures

------------------------------
## 📂 Offline Assets & Patches
To ensure 100% offline reliability, the following modifications have been implemented:

   1. WebSocket Defaulting: src/app.js is patched to connect to ws://localhost:8765/v1/realtime out of the box.
   2. Offline Typography: All external Google Fonts have been replaced with local @font-face definitions and binaries in public/vendor/.
   3. Local LFS Binaries: brunette.glb (3D model) and model-en-mixed.bin (viseme classifier) are included locally.
   4. Frozen Dependencies: Uses bun install --frozen-lockfile to ensure environment stability.
   5. Orchestration Suite: Includes specialized scripts for launching Ollama, the speech server, and the frontend simultaneously.

------------------------------
## 🚀 Execution & Usage## ⚡ The One-Liner (Fast Start)
Once everything is properly downloaded and cached, you can launch the entire environment with a simple one-liner:

cd /home/user/gemma-avatar && OLLAMA_MODEL=gemma4:26b PATH="$HOME/.bun/bin:$PATH" ./run-all-local.sh

Then simply open Microsoft Edge or Chrome to: http://localhost:3000
Why this works:

* Native Overrides: Key scripts (run-all-local.sh, start-ollama.sh, start-speech-to-speech.sh) support model overrides via the OLLAMA_MODEL variable.
* Automatic Fallbacks: If the variable is unset, the launcher intelligently picks from available Gemma models in a pre-defined fallback order.
* Efficiency: Using the 26b model via this command is typically faster and lighter on system resources than the 31b default.

------------------------------
## 🛠️ Detailed Setup (WSL2 / Linux)## 1. Initial Model Acquisition (One-time Online Step)
Before going offline, pull the necessary models and assets:

# Pull LLM via Ollama
ollama pull gemma4:31b
# Setup speech environment and download TTS assets
pip install -r gemma-avatar-s2s-requirements-lock.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/parakeet-tdt-1.1b')"
/home/user/venvs/gemma-avatar-s2s/bin/python download_tts_assets.py

## 2. Manual Service Launch
If you prefer to run services in separate terminals for debugging:

* LLM: ./start-ollama.sh
* Speech Server: ./start-speech-to-speech.sh
* Web UI: ./start-avatar-frontend.sh

------------------------------
## 🧪 Testing & Verification

* Synthetic Mic Test: Access http://localhost:3000/?fakemic=1 and execute getClient().requestResponse() in the console.
* Tool-Calling: Run ./test-tools.sh to verify make_hand_gesture function calls via Ollama.
* Barge-in Support: Test interruptions while the avatar is speaking; audio and mouth movements should settle immediately upon new input.

## 💾 Backup & Portability
To preserve this environment as a single portable archive:

# From Windows PowerShell
wsl --shutdown
wsl --export Ubuntu path\to\your\backup\gemma-avatar-wsl.tar

## 🤝 Credits

* Victor: Original Hugging Face Space foundation.
* Models: NVIDIA (Parakeet), Silero (VAD), Qwen Team (TTS), Google (Gemma).
* Tools: Three.js, Bun, and Ollama for the local runtime orchestration.

------------------------------
Built over a 6-day sprint to liberate high-fidelity voice AI from the cloud.

------------------------------
Currently in the works:

*Real-Time Collaborative Co-Programming:* A shared, synchronized canvas for simultaneous, multi-user editing and design.

*Introspective Live-State Debugging:* The ability for the assistant to perform real-time inspection and explanation of its own active source
code and running processes.

*Integrated Sandbox and Terminal Environments:* Localized, execution-ready workspaces for testing, running, and
verifying code snippets instantly.

*Proactive Agentic Intelligence:* An AI layer capable of predictive reasoning to anticipate needs before they are explicitly
stated.

*Predictive Resource Provisioning:* The automated preparation of tools, environments, or data streams based on the trajectory of
the current task.

*Closed-Loop Feedback Systems:* A continuous cycle of execution, observation, and autonomous refinement between the human
and the AI.
