"""XTTS-v2 (Coqui TTS) handler for the Gemma Avatar speech-to-speech pipeline.

The upstream `speech-to-speech` server ships handlers for Qwen3-TTS, Kokoro,
ChatTTS, Facebook MMS and Pocket TTS. This module adds a drop-in handler for
fine-tuned **Coqui XTTS-v2** checkpoints (the ``model.pth`` / ``config.json`` /
``vocab.json`` + reference-wav layout that most XTTS fine-tuning tools produce),
so the avatar can speak with your own cloned voice.

How it plugs in
---------------
`local_s2s_launcher.py` monkeypatches
`speech_to_speech.s2s_pipeline.get_tts_handler` so that ``--tts xtts`` (or
``GEMMA_TTS_ENGINE=xtts``) returns :class:`XttsTTSHandler` instead of the stock
Qwen3 handler. Everything else in the pipeline (VAD -> STT -> LLM -> TTS ->
realtime websocket) is untouched.

Audio contract
--------------
The pipeline moves PCM16 mono audio at 16 kHz. XTTS-v2 synthesises at 24 kHz,
so every chunk is resampled with a polyphase filter and quantised to int16
before it is yielded, in ``blocksize``-sample chunks (default 512 = 32 ms).

Configuration
-------------
All settings can be passed to :meth:`XttsTTSHandler.setup` or, which is what the
launcher does, through ``XTTS_*`` environment variables (see
:func:`setup_kwargs_from_env`). Run ``python check_xtts_voice.py <voice-dir>``
to validate a voice folder before starting the stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from sys import platform
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    EndOfResponse,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from xtts_engine import (
    DEFAULT_BLOCKSIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHARS,
    VOICE_AUDIO_SUFFIXES,
    VOICE_REF_CANDIDATES,
    clean_text,
    resolve_language,
    setup_kwargs_from_env,
    split_for_xtts,
)

logger = logging.getLogger(__name__)

PIPELINE_SR = 16000

__all__ = ["XttsTTSHandler", "setup_kwargs_from_env", "PIPELINE_SR"]


class XttsTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Text-to-speech stage backed by a Coqui XTTS-v2 checkpoint."""

    # ── setup ────────────────────────────────────────────────────────────
    def setup(
        self,
        should_listen: Event,
        voice_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        speaker_wav: str | Path | None = None,
        language: str = DEFAULT_LANGUAGE,
        device: str = "auto",
        dtype: str = "fp32",
        speed: float = 1.0,
        blocksize: int = DEFAULT_BLOCKSIZE,
        stream: bool = True,
        stream_chunk_size: int = 20,
        overlap_wav_len: int = 1024,
        temperature: float = 0.75,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 50,
        top_p: float = 0.85,
        do_sample: bool = True,
        enable_text_splitting: bool = False,
        gpt_cond_len: int = 6,
        max_ref_len: int = 30,
        sound_norm_refs: bool = False,
        max_chars: int = DEFAULT_MAX_CHARS,
        warmup: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.language = resolve_language(language)
        self.speed = speed
        self.blocksize = max(1, int(blocksize))
        self.stream = bool(stream)
        self.stream_chunk_size = max(1, int(stream_chunk_size))
        self.overlap_wav_len = max(0, int(overlap_wav_len))
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = int(top_k)
        self.top_p = top_p
        self.do_sample = bool(do_sample)
        self.enable_text_splitting = bool(enable_text_splitting)
        self.gpt_cond_len = max(1, int(gpt_cond_len))
        self.max_ref_len = max(1, int(max_ref_len))
        self.sound_norm_refs = bool(sound_norm_refs)
        self.max_chars = int(max_chars)
        self.gen_kwargs = dict(gen_kwargs or {})

        paths = self._resolve_voice_files(
            voice_dir=voice_dir,
            model_path=model_path,
            config_path=config_path,
            vocab_path=vocab_path,
            speaker_wav=speaker_wav,
        )
        self.voice_dir = paths["voice_dir"]
        self.model_path = paths["model_path"]
        self.config_path = paths["config_path"]
        self.vocab_path = paths["vocab_path"]
        self.speaker_wav = paths["speaker_wav"]

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype, self.device)

        logger.info(
            "Loading XTTS voice '%s' (model=%s, device=%s, dtype=%s)",
            self.voice_dir.name if self.voice_dir else self.model_path.parent.name,
            self.model_path.name,
            self.device,
            self.dtype,
        )
        self.model = self._load_model()
        self.output_sample_rate = int(getattr(self.model.args, "output_sample_rate", 24000) or 24000)
        logger.info("XTTS loaded: output sample rate %d Hz -> pipeline %d Hz", self.output_sample_rate, PIPELINE_SR)

        # Default voice identity. Conditioning latents are expensive enough that we
        # compute them once at startup and keep them for the whole process.
        self._latents_cache: dict[str, tuple[Any, Any]] = {}
        self.default_voice_key = str(self.speaker_wav)
        self.gpt_cond_latent, self.speaker_embedding = self._conditioning_latents(self.speaker_wav)
        self.voice_name = self.speaker_wav.stem

        if warmup:
            self._warmup()

    # ── configuration helpers ────────────────────────────────────────────
    @staticmethod
    def _resolve_voice_files(
        *,
        voice_dir: str | Path | None,
        model_path: str | Path | None,
        config_path: str | Path | None,
        vocab_path: str | Path | None,
        speaker_wav: str | Path | None,
    ) -> dict[str, Any]:
        """Locate the checkpoint + reference clip, failing early with a useful message."""
        resolved_dir: Optional[Path] = None
        if voice_dir:
            resolved_dir = Path(voice_dir).expanduser()
            if not resolved_dir.is_dir():
                raise FileNotFoundError(
                    f"XTTS voice directory not found: {resolved_dir}. "
                    "Point XTTS_VOICE_DIR (or --xtts-voice-dir) at the folder that holds model.pth."
                )

        model = Path(model_path).expanduser() if model_path else (resolved_dir / "model.pth" if resolved_dir else None)
        if model is None:
            raise ValueError(
                "XTTS needs a checkpoint: set XTTS_VOICE_DIR to a folder containing model.pth, "
                "or set XTTS_MODEL_PATH explicitly."
            )
        if resolved_dir is None:
            resolved_dir = model.parent

        config = Path(config_path).expanduser() if config_path else resolved_dir / "config.json"
        vocab = Path(vocab_path).expanduser() if vocab_path else resolved_dir / "vocab.json"

        if speaker_wav:
            ref = Path(speaker_wav).expanduser()
        else:
            ref = None
            for candidate in VOICE_REF_CANDIDATES:
                maybe = resolved_dir / candidate
                if maybe.is_file():
                    ref = maybe
                    break
            if ref is None:
                audio = sorted(p for p in resolved_dir.glob("*") if p.suffix.lower() in VOICE_AUDIO_SUFFIXES)
                if audio:
                    ref = audio[0]
            if ref is None:
                raise FileNotFoundError(
                    f"No reference audio found in {resolved_dir}. XTTS clones from a reference clip: "
                    "add speaker_ref.wav (3-30 s of clean speech) or set XTTS_SPEAKER_WAV."
                )

        missing = [str(p) for p in (model, config, vocab, ref) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "XTTS voice files are missing: " + ", ".join(missing) + ". "
                "Expected model.pth, config.json, vocab.json and a reference wav in the voice folder."
            )

        return {
            "voice_dir": resolved_dir,
            "model_path": model,
            "config_path": config,
            "vocab_path": vocab,
            "speaker_wav": ref,
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = (device or "auto").strip().lower()
        import torch

        if requested in ("auto", ""):
            if torch.cuda.is_available():
                return "cuda"
            if platform == "darwin" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("XTTS_DEVICE=%s requested but CUDA is unavailable; falling back to cpu", requested)
            return "cpu"
        if requested == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            logger.warning("XTTS_DEVICE=mps requested but MPS is unavailable; falling back to cpu")
            return "cpu"
        return requested

    @staticmethod
    def _resolve_dtype(dtype: str, device: str) -> str:
        requested = (dtype or "fp32").strip().lower()
        if requested not in ("fp32", "fp16", "bf16"):
            logger.warning("Unknown XTTS_DTYPE=%s; using fp32", requested)
            requested = "fp32"
        if requested != "fp32" and not device.startswith("cuda"):
            logger.warning("XTTS_DTYPE=%s only helps on CUDA; using fp32 on %s", requested, device)
            return "fp32"
        return requested

    def _load_model(self) -> Any:
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError(
                "Coqui TTS is required for --tts xtts. Install it in the speech-to-speech venv with:\n"
                "  pip install -r requirements-xtts.txt\n"
                "(or: pip install 'coqui-tts>=0.28.0')"
            ) from exc

        import torch

        config = XttsConfig()
        config.load_json(str(self.config_path))
        model = Xtts.init_from_config(config)
        # `load_checkpoint` also wires up vocab.json and, when present,
        # speakers_xtts.pth (optional for reference-clip cloning).
        model.load_checkpoint(
            config,
            checkpoint_path=str(self.model_path),
            vocab_path=str(self.vocab_path),
            eval=True,
        )

        if self.dtype == "fp16":
            model = model.half()
        elif self.dtype == "bf16":
            model = model.bfloat16()
        model.to(self.device)
        model.eval()

        if self.device.startswith("cuda"):
            logger.info(
                "XTTS on CUDA (%s), %.2f GiB allocated",
                torch.cuda.get_device_name(0),
                torch.cuda.memory_allocated() / 1024**3,
            )
        return model

    def _conditioning_latents(self, ref_path: Path) -> tuple[Any, Any]:
        """Compute (and cache) the GPT conditioning latents for one reference clip."""
        key = str(ref_path)
        cached = self._latents_cache.get(key)
        if cached is not None:
            return cached

        import torch

        start = perf_counter()
        logger.info("Computing XTTS speaker latents from %s", ref_path)
        with torch.inference_mode():
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[str(ref_path)],
                max_ref_length=self.max_ref_len,
                gpt_cond_len=self.gpt_cond_len,
                gpt_cond_chunk_len=min(6, self.gpt_cond_len),
                sound_norm_refs=self.sound_norm_refs,
            )
        self._latents_cache[key] = (gpt_cond_latent, speaker_embedding)
        logger.info("XTTS speaker latents ready in %.2fs", perf_counter() - start)
        return gpt_cond_latent, speaker_embedding

    def _warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        start = perf_counter()
        try:
            for _ in self._iter_audio_chunks("Hello there.", self.language):
                pass
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning("XTTS warmup failed (continuing anyway): %s", exc)
        else:
            logger.info("XTTS warmed up in %.2fs", perf_counter() - start)

    # ── voice selection ──────────────────────────────────────────────────
    def _voice_candidates(self, voice: str) -> list[Path]:
        """Map a requested voice name onto a reference clip on disk."""
        raw = voice.strip()
        if not raw:
            return []
        candidates: list[Path] = []

        direct = Path(raw).expanduser()
        if direct.is_file():
            candidates.append(direct)

        if self.voice_dir is not None:
            base = Path(raw).name
            if not base.lower().endswith(VOICE_AUDIO_SUFFIXES):
                for suffix in (".wav", ".flac", ".mp3"):
                    candidates.append(self.voice_dir / f"{base}{suffix}")
                    candidates.append(self.voice_dir / "voices" / f"{base}{suffix}")
            else:
                candidates.append(self.voice_dir / base)
                candidates.append(self.voice_dir / "voices" / base)
        return candidates

    def _apply_session_voice(self, runtime_config: Any, response: Any) -> None:
        """Honour a per-session `voice` field when it points at a reference clip."""
        requested: Optional[str] = None
        if response is not None and getattr(response, "audio", None):
            output = getattr(response.audio, "output", None)
            if output is not None and getattr(output, "voice", None):
                requested = str(output.voice)
        if not requested and runtime_config is not None:
            audio_cfg = getattr(getattr(runtime_config, "session", None), "audio", None)
            output_cfg = getattr(audio_cfg, "output", None) if audio_cfg is not None else None
            if output_cfg is not None and getattr(output_cfg, "voice", None):
                requested = str(output_cfg.voice)

        if not requested:
            return
        for candidate in self._voice_candidates(requested):
            if not candidate.is_file():
                continue
            if str(candidate) == self.default_voice_key:
                self.gpt_cond_latent, self.speaker_embedding = self._latents_cache[self.default_voice_key]
                self.voice_name = candidate.stem
                return
            self.gpt_cond_latent, self.speaker_embedding = self._conditioning_latents(candidate)
            self.voice_name = candidate.stem
            logger.info("XTTS voice switched to %s", candidate)
            return

    # ── pipeline entry point ─────────────────────────────────────────────
    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)

        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text, language_code = self._coalesce_pending_tts_input(tts_input)
        text = clean_text(text)
        if not text:
            logger.debug("Skipping empty TTS text for turn=%s", tts_input.turn_id)
            return

        language = resolve_language(language_code, self.language)
        self._apply_session_voice(getattr(tts_input, "runtime_config", None), getattr(tts_input, "response", None))

        logger.debug("XTTS: turn=%s language=%s chars=%d", tts_input.turn_id, language, len(text))
        logger.info("ASSISTANT: %s", text)

        first_audio = True
        try:
            for chunk in self._iter_audio_chunks(text, language):
                if first_audio:
                    self._log_first_audio_latency(tts_input)
                    first_audio = False
                yield chunk
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("XTTS synthesis failed: %s", exc, exc_info=True)

    def _log_first_audio_latency(self, tts_input: TTSInput) -> None:
        speech_stopped_at_s = getattr(tts_input, "speech_stopped_at_s", None)
        if speech_stopped_at_s is None:
            return
        latency_s = perf_counter() - speech_stopped_at_s
        if latency_s < 0:
            return
        logger.info(
            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
            latency_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def _coalesce_pending_tts_input(self, current_input: TTSInput) -> tuple[str, Optional[str]]:
        """Merge queued sentence fragments into one synthesis call (better prosody, less overhead)."""
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return current_input.text, current_input.language_code

        text = current_input.text
        language_code = current_input.language_code
        parts = [text.strip()] if text and text.strip() else []

        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if is_control_message(next_item, SESSION_END.kind):
                    break
                if isinstance(next_item, bytes) and next_item == PIPELINE_END:
                    break
                if isinstance(next_item, EndOfResponse):
                    break
                if not isinstance(next_item, TTSInput):
                    break
                if current_input.turn_id != next_item.turn_id or current_input.turn_revision != next_item.turn_revision:
                    break
                if (
                    language_code is not None
                    and next_item.language_code is not None
                    and next_item.language_code != language_code
                ):
                    break
                self.queue_in.queue.popleft()
                if next_item.text.strip():
                    parts.append(next_item.text.strip())
                if language_code is None:
                    language_code = next_item.language_code

        return " ".join(parts).strip(), language_code

    # ── audio generation ─────────────────────────────────────────────────
    def _synthesize(self, text: str, language: str) -> Iterator[Any]:
        """Yield raw 24 kHz float audio pieces from XTTS (streamed or one-shot)."""
        import torch

        kwargs = {
            "temperature": self.temperature,
            "length_penalty": float(self.length_penalty),
            "repetition_penalty": float(self.repetition_penalty),
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "speed": self.speed,
            "enable_text_splitting": self.enable_text_splitting,
            **self.gen_kwargs,
        }

        with torch.inference_mode():
            if self.stream:
                yield from self.model.inference_stream(
                    text=text,
                    language=language,
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    stream_chunk_size=self.stream_chunk_size,
                    overlap_wav_len=self.overlap_wav_len,
                    **kwargs,
                )
            else:
                result = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    **kwargs,
                )
                yield result["wav"]

    def _iter_audio_chunks(self, text: str, language: str) -> Iterator[np.ndarray]:
        """Synthesise `text` and yield int16 16 kHz chunks of exactly `blocksize` samples."""
        from scipy.signal import resample_poly

        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        start = perf_counter()
        total_samples = 0
        first_piece = True
        found_speech = False
        leftover = np.array([], dtype=np.int16)

        for piece in split_for_xtts(text, self.max_chars):
            for raw_audio in self._synthesize(piece, language):
                if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                    logger.info("XTTS generation cancelled (interruption)")
                    return

                audio = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
                if audio.size == 0:
                    continue

                if first_piece:
                    logger.debug("XTTS first audio piece after %.2fs", perf_counter() - start)
                    first_piece = False

                if self.output_sample_rate != PIPELINE_SR:
                    gcd = int(np.gcd(PIPELINE_SR, self.output_sample_rate))
                    audio = resample_poly(
                        audio,
                        up=PIPELINE_SR // gcd,
                        down=self.output_sample_rate // gcd,
                    ).astype(np.float32, copy=False)

                int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

                # Drop the silent generator ramp-up once, keeping a little preroll so
                # soft initial phonemes are not clipped.
                if not found_speech:
                    threshold = int(32768 * 0.01)
                    above = np.abs(int16) > threshold
                    if not np.any(above):
                        continue
                    head = max(0, int(np.argmax(above)) - int(PIPELINE_SR * 0.040))
                    int16 = int16[head:]
                    found_speech = True

                int16 = np.concatenate([leftover, int16]) if leftover.size else int16
                usable = (len(int16) // self.blocksize) * self.blocksize
                for offset in range(0, usable, self.blocksize):
                    if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                        logger.info("XTTS generation cancelled (interruption)")
                        return
                    yield int16[offset : offset + self.blocksize]
                    total_samples += self.blocksize
                leftover = int16[usable:]

        if leftover.size:
            yield np.pad(leftover, (0, self.blocksize - len(leftover)))
            total_samples += len(leftover)

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0.0
        logger.info(
            "XTTS generated %.2fs audio in %.2fs (RTF: %.2f)",
            audio_duration,
            generation_time,
            rtf,
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    def on_session_end(self) -> None:
        default = self._latents_cache.get(self.default_voice_key)
        if default is not None:
            self.gpt_cond_latent, self.speaker_embedding = default
            self.voice_name = self.speaker_wav.stem
        logger.debug("XTTS session state reset")
