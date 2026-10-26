# 🎙️ Speak with your own voice: XTTS-v2 engine for Gemma Avatar

Short answer: **yes, you can swap the TTS stage.** Gemma Avatar's speech-to-speech
server talks to the TTS model through a queue of text → PCM16 audio, so any engine
that can turn a sentence into 16 kHz PCM can drive the avatar (and its lip-sync).

This repo now ships a handler for **Coqui XTTS-v2 fine-tunes** — which is exactly
what your folder is:

```text
C:\Users\l3ung\forge\voices\my-voice\
  model.pth        # XTTS-v2 GPT fine-tune checkpoint (~1.8 GB)
  config.json      # XttsConfig for that checkpoint
  vocab.json       # BPE tokenizer vocabulary
  speaker_ref.wav  # reference clip → this is what makes it *your* voice
  voice.json       # metadata written by the export tool; not needed at runtime
```

`model.pth` + `config.json` + `vocab.json` identify the checkpoint; `speaker_ref.wav`
provides the speaker conditioning (XTTS always clones from a reference clip, even
with a fine-tuned checkpoint). `voice.json` is informational and ignored — the
handler does not require it, but `check_xtts_voice.py` will print what it contains.

> **Note on other engines:** the stock server also supports `kokoro`, `pocket`,
> `chatTTS` and `facebookMMS`, but none of them load your `model.pth` — that is an
> XTTS-specific format. XTTS-v2 is the engine that matches your files.

---

## 1. Install Coqui TTS in the speech-to-speech venv

The XTTS engine is optional, so it is **not** in `gemma-avatar-s2s-requirements-lock.txt`
(the default Qwen3-TTS path stays untouched).

```bash
source ~/venvs/gemma-avatar-s2s/bin/activate
pip install -r requirements-xtts.txt          # coqui-tts>=0.28.0
```

`torch`, `torchaudio`, `transformers>=4.57` and `scipy` are already pinned by the
main lock file, so this only adds the Coqui inference stack (librosa, soundfile, …).

---

## 2. Make the voice folder visible to WSL

Either copy it in (fastest; Windows paths under `/mnt/c` are slow to read):

```bash
mkdir -p ~/voices
cp -r "/mnt/c/Users/l3ung/forge/voices/my-voice" ~/voices/
```

…or point at the Windows folder directly:

```bash
export XTTS_VOICE_DIR="/mnt/c/Users/l3ung/forge/voices/my-voice"
```

---

## 3. Validate the folder (no GPU, no model load)

```bash
python check_xtts_voice.py ~/voices/my-voice
```

It checks the four required files, reads the checkpoint config (model type,
languages, output sample rate), inspects `speaker_ref.wav` (length, channels,
sample rate) and prints the exact command to launch with. Exit code `0` = ready.

Typical output:

```text
XTTS voice folder: /home/l3ung/voices/my-voice
  [ok]   model.pth            1.8 GiB
  [ok]   config.json          4.2 KiB
  [ok]   vocab.json           1.1 MiB
  [ok]   voice.json           312 B (optional)
  config: model='xtts' languages=17 output_sample_rate=24000
  [ok]   speaker_ref.wav      22050 Hz, 1 ch, 7.42s
Result: ready for XTTS.
```

---

## 4. Run the stack with your voice

One command (all three services, same as before):

```bash
TTS_ENGINE=xtts XTTS_VOICE_DIR="$HOME/voices/my-voice" \
  OLLAMA_MODEL=gemma4:26b PATH="$HOME/.bun/bin:$PATH" ./run-all-local.sh
```

Or start only the audio server (useful for debugging TTS):

```bash
TTS_ENGINE=xtts XTTS_VOICE_DIR="$HOME/voices/my-voice" ./start-speech-to-speech.sh
```

`./start-speech-to-speech.sh qwen3` (or unsetting `TTS_ENGINE`) switches back to the
bundled Qwen3-TTS voice — the two engines are selected at startup, everything else
(LLM, STT, avatar, lip-sync) is identical.

The first launch logs look like:

```text
TTS engine: xtts | voice dir: /home/l3ung/voices/my-voice | speaker: <auto>
Loading XTTS voice 'my-voice' (model=model.pth, device=cuda, dtype=fp32)
XTTS loaded: output sample rate 24000 Hz -> pipeline 16000 Hz
Computing XTTS speaker latents from .../speaker_ref.wav
XTTS speaker latents ready in 1.84s
XTTS warmed up in 3.10s
```

---

## 5. Tuning (`XTTS_*` environment variables)

Everything is optional; defaults are tuned for a GPU build.

| Variable | Default | What it does |
| --- | --- | --- |
| `XTTS_VOICE_DIR` | – | Folder with `model.pth`/`config.json`/`vocab.json` + reference wav |
| `XTTS_MODEL_PATH` / `XTTS_CONFIG_PATH` / `XTTS_VOCAB_PATH` / `XTTS_SPEAKER_WAV` | derived from `XTTS_VOICE_DIR` | Override individual files |
| `XTTS_LANGUAGE` | `en` | Default XTTS language; per-utterance language detection still applies when available |
| `XTTS_DEVICE` | `auto` | `auto`, `cuda`, `cuda:1`, `cpu`, `mps` |
| `XTTS_DTYPE` | `fp32` | `fp16` / `bf16` (CUDA only) — smaller VRAM, slightly riskier |
| `XTTS_STREAM` | `1` | Stream partial audio from XTTS instead of waiting for a whole sentence |
| `XTTS_STREAM_CHUNK_SIZE` | `20` | XTTS codec tokens per streamed chunk; lower = earlier first audio |
| `XTTS_OVERLAP_WAV_LEN` | `1024` | Cross-fade overlap between streamed chunks (samples) |
| `XTTS_SPEED` | `1.0` | Speech rate multiplier |
| `XTTS_TEMPERATURE` / `XTTS_TOP_P` / `XTTS_TOP_K` | `0.75` / `0.85` / `50` | Sampling; lower temperature = steadier, less expressive |
| `XTTS_REPETITION_PENALTY` / `XTTS_LENGTH_PENALTY` | `10.0` / `1.0` | Curb loops / lengthen speech |
| `XTTS_MAX_CHARS` | `400` | Text is split on sentence boundaries above this (XTTS caps text tokens) |
| `XTTS_GPT_COND_LEN` | `6` | Seconds of reference clip used for GPT conditioning |
| `XTTS_MAX_REF_LEN` | `30` | Max seconds read from each reference clip |
| `XTTS_SOUND_NORM_REFS` | `0` | `1` = normalise quiet references |
| `XTTS_BLOCKSIZE` | `512` | Output chunk size in samples (512 = 32 ms @ 16 kHz) |
| `XTTS_WARMUP` | `1` | Run one throwaway synthesis at startup |

The same settings exist as CLI options on the launcher
(`--xtts-voice-dir`, `--xtts-language`, `--xtts-device`, …):

```bash
python ./local_s2s_launcher.py --mode realtime --tts xtts \
  --xtts-voice-dir "$HOME/voices/my-voice" --xtts-temperature 0.65 ...
```

---

## 6. Switching voices at runtime

The pipeline sends a per-session `session.audio.output.voice` string. If that
string resolves to a reference clip (absolute path, `<name>.wav` inside the voice
folder, or `<voice-dir>/voices/<name>.wav`), the handler recomputes conditioning
for it, caches it, and speaks with that voice. So you can park extra clips in
`~/voices/my-voice/voices/` and pick one per session without restarting.

---

## 7. What changed in this repo

| File | Role |
| --- | --- |
| `xtts_tts_handler.py` | `XttsTTSHandler`: loads the checkpoint, computes latents, streams/resamples to PCM16 @ 16 kHz, honours barge-in |
| `xtts_engine.py` | Dependency-free config layer: `--tts xtts` / `--xtts-*` → `XTTS_*` env vars, language mapping, text cleanup/splitting |
| `local_s2s_launcher.py` | Patches `parse_arguments` + `get_tts_handler` so `tts="xtts"` builds `XttsTTSHandler`; passes the pipeline's `cancel_scope`/`speculative_turns` into it |
| `start-speech-to-speech.sh` | `TTS_ENGINE=xtts` (or first CLI arg) selects the engine, validates the voice folder, keeps Qwen3 defaults intact |
| `check_xtts_voice.py` | Stdlib-only voice-folder validator |
| `requirements-xtts.txt` | `coqui-tts>=0.28.0` |
| `tests/test_xtts_handler.py` | 21 offline tests (audio contract, cancellation, CLI/env plumbing, launcher dispatch) |

The audio contract is unchanged: the handler yields **int16 mono chunks, exactly
512 samples (32 ms), 16 kHz**, which is what the realtime websocket, the playback
worklet and HeadAudio's viseme classifier already expect. No frontend changes.

---

## 8. Expectations: latency and quality

- **Streaming**: XTTS is an autoregressive GPT + HiFi-GAN decoder. With
  `XTTS_STREAM=1` you get the first audio after ~15–25 codec tokens
  (≈0.3–0.8 s on an RTX 3060-class GPU, ~1–2 s on CPU), then audio keeps flowing
  while the sentence is still being generated. Without streaming, the whole
  sentence is generated first (adds ~0.5–1.5 s of dead air).
- **Throughput**: roughly `RTF ≈ 0.2–0.4` on a modern GPU, `≥ 1.0` on CPU. The
  pipeline synthesises per sentence, so long answers stay responsive.
- **Quality**: fine-tuned XTTS checkpoints reproduce timbre well but are a bit more
  prone to odd pronunciations than Qwen3-TTS. Good reference clips (6–12 s, clean,
  single speaker, no music) and `XTTS_TEMPERATURE=0.65` usually help.
- **License**: XTTS-v2 weights and fine-tunes are under the Coqui Public Model
  License (non-commercial by default). Check it before shipping anything public.

---

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ERROR: Coqui TTS is not installed` | `pip install -r requirements-xtts.txt` inside `~/venvs/gemma-avatar-s2s` |
| `XTTS voice directory not found` | WSL cannot see the Windows path — copy the folder into `~/voices` or use `/mnt/c/...` |
| `No reference audio found` | Need `speaker_ref.wav` (or `reference.wav`) in the folder, or set `XTTS_SPEAKER_WAV` |
| `No module named 'TTS'` after install | You installed into a different Python; re-activate the venv (`VENV_PATH=...`) |
| `UnpicklingError` / `weights_only` on load | Old `TTS` 0.22 checkpoints + `torch>=2.6`; use `coqui-tts>=0.28` (handled here) |
| `AssertionError: text ... maximum of 400 tokens` | Lower `XTTS_MAX_CHARS` (300 is safe) |
| Robotic / clipped first word | Increase `XTTS_OVERLAP_WAV_LEN`, or `XTTS_STREAM=0` to compare |
| Long silence at the start of every reply | Keep `XTTS_STREAM=1`; the handler trims the initial silent ramp |
| CUDA OOM | `XTTS_DTYPE=fp16`, or run the LLM/TTS on different GPUs (`XTTS_DEVICE=cuda:1`) |
| Avatar mouth out of sync | TTS output must stay 16 kHz PCM16 — verify you did not disable the resample path or `XTTS_BLOCKSIZE` sane |

Run the offline test suite any time:

```bash
python -m unittest discover -s tests -v
```
