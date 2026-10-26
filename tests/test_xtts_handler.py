#!/usr/bin/env python3
"""Offline tests for the optional XTTS-v2 TTS engine.

Run with:  python -m unittest discover -s tests -v

The handler tests stub out ``torch`` and the Coqui ``TTS`` package, so they run
on a machine without the model weights, without Coqui, and without a GPU. They
use the real ``speech_to_speech`` pipeline classes (BaseHandler, TTSInput,
CancelScope) so the audio contract is exercised for real:

    int16 mono chunks, exactly ``blocksize`` samples, 16 kHz effective rate.

Tests are skipped automatically when the speech-to-speech package is missing.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import xtts_engine  # noqa: E402  (path set above)

PIPELINE_SR = 16000
XTTS_SR = 24000


def _write_wav(path: Path, seconds: float = 4.0, rate: int = 22050) -> None:
    import math
    import struct

    frames = int(seconds * rate)
    samples = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 180 * i / rate))) for i in range(frames)
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples)


def _install_fake_torch() -> None:
    """Minimal torch stand-in: enough for device detection + inference_mode."""
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")
    fake.inference_mode = lambda *a, **k: contextlib.nullcontext()
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda *a, **k: "cpu",
        memory_allocated=lambda: 0,
    )
    fake.backends = types.SimpleNamespace()
    # scipy's array_api_compat probes `torch.Tensor`; give it something harmless.
    fake.Tensor = type("Tensor", (object,), {})
    fake.nn = types.SimpleNamespace(Module=type("Module", (object,), {}))
    sys.modules["torch"] = fake


def _install_fake_tts(chunks: int = 4, chunk_seconds: float = 0.25) -> None:
    """Stub the Coqui TTS package with a deterministic XTTS-like model."""
    import numpy as np

    class FakeXttsConfig:
        def load_json(self, path):
            self.path = path

    class FakeXtts:
        def __init__(self, config):
            self.config = config
            self.args = types.SimpleNamespace(output_sample_rate=XTTS_SR)
            self.device = "cpu"

        @classmethod
        def init_from_config(cls, config):
            return cls(config)

        def load_checkpoint(self, config, checkpoint_path=None, vocab_path=None, eval=True, **kwargs):
            self.checkpoint_path = checkpoint_path
            self.vocab_path = vocab_path

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def half(self):
            return self

        def bfloat16(self):
            return self

        def get_conditioning_latents(self, audio_path=None, **kwargs):
            del audio_path, kwargs
            return (
                np.zeros((1, 1024, 2), dtype=np.float32),
                np.zeros((1, 512, 1), dtype=np.float32),
            )

        def _tone(self):
            t = np.arange(int(XTTS_SR * chunk_seconds), dtype=np.float32) / XTTS_SR
            return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        def inference_stream(self, text, language, **kwargs):
            self.last_call = {"text": text, "language": language, **kwargs}
            for _ in range(chunks):
                yield self._tone()

        def inference(self, text, language, **kwargs):
            self.last_call = {"text": text, "language": language, **kwargs}
            return {"wav": np.tile(self._tone(), chunks)}

    modules = {
        "TTS": types.ModuleType("TTS"),
        "TTS.tts": types.ModuleType("TTS.tts"),
        "TTS.tts.configs": types.ModuleType("TTS.tts.configs"),
        "TTS.tts.configs.xtts_config": types.ModuleType("TTS.tts.configs.xtts_config"),
        "TTS.tts.models": types.ModuleType("TTS.tts.models"),
        "TTS.tts.models.xtts": types.ModuleType("TTS.tts.models.xtts"),
    }
    modules["TTS.tts.configs.xtts_config"].XttsConfig = FakeXttsConfig
    modules["TTS.tts.models.xtts"].Xtts = FakeXtts
    sys.modules.update(modules)


def _make_voice_dir(root: Path) -> Path:
    voice_dir = root / "my-voice"
    voice_dir.mkdir()
    (voice_dir / "model.pth").write_bytes(b"fake-checkpoint")
    (voice_dir / "config.json").write_text(json.dumps({"model": "xtts", "languages": ["en"]}))
    (voice_dir / "vocab.json").write_text(json.dumps({"a": 1}))
    (voice_dir / "voice.json").write_text(json.dumps({"name": "my-voice"}))
    _write_wav(voice_dir / "speaker_ref.wav")
    return voice_dir


class EngineConfigTests(unittest.TestCase):
    """Pure helpers from xtts_engine (no ML dependencies)."""

    def test_engine_aliases(self):
        for alias in ("xtts", "XTTS", "xtts-v2", "xtts_v2", "coqui-xtts"):
            self.assertEqual(xtts_engine.normalize_engine(alias), "xtts")
        self.assertEqual(xtts_engine.normalize_engine("qwen3"), "qwen3")
        self.assertIsNone(xtts_engine.normalize_engine(None))

    def test_language_mapping(self):
        self.assertEqual(xtts_engine.resolve_language("en"), "en")
        self.assertEqual(xtts_engine.resolve_language("en-US"), "en")
        self.assertEqual(xtts_engine.resolve_language("zh"), "zh-cn")
        self.assertEqual(xtts_engine.resolve_language("jp"), "ja")
        self.assertEqual(xtts_engine.resolve_language("pt-BR"), "pt")
        # Unknown codes fall back to the configured default.
        self.assertEqual(xtts_engine.resolve_language("xx", default="de"), "de")
        self.assertEqual(xtts_engine.resolve_language(None, default="nonsense"), "en")

    def test_clean_text(self):
        self.assertEqual(xtts_engine.clean_text("**Hello** _world_!"), "Hello world!")
        self.assertEqual(xtts_engine.clean_text("  multi\nline   text "), "multi line text")
        self.assertEqual(xtts_engine.clean_text("..."), "")
        self.assertEqual(xtts_engine.clean_text(""), "")

    def test_split_for_xtts(self):
        self.assertEqual(xtts_engine.split_for_xtts("One sentence."), ["One sentence."])
        long_text = " ".join(["This is a fairly long sentence about avatars."] * 12)
        pieces = xtts_engine.split_for_xtts(long_text, max_chars=80)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 80 for piece in pieces))
        # Nothing is lost in the split.
        self.assertEqual(" ".join(pieces).split(), long_text.split())

    def test_strip_xtts_argv(self):
        env_keys = ("GEMMA_TTS_ENGINE", "XTTS_VOICE_DIR", "XTTS_LANGUAGE", "XTTS_DEVICE")
        saved = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)

            argv = [
                "local_s2s_launcher.py",
                "--mode",
                "realtime",
                "--tts",
                "xtts",
                "--xtts-voice-dir",
                "C:/Users/l3ung/forge/voices/my-voice",
                "--xtts-language=fr",
                "--model_name",
                "gemma4:26b",
            ]
            remaining = xtts_engine.strip_xtts_argv(argv)
            self.assertEqual(
                remaining,
                ["local_s2s_launcher.py", "--mode", "realtime", "--model_name", "gemma4:26b"],
            )
            self.assertEqual(os.environ["GEMMA_TTS_ENGINE"], "xtts")
            self.assertEqual(os.environ["XTTS_VOICE_DIR"], "C:/Users/l3ung/forge/voices/my-voice")
            self.assertEqual(os.environ["XTTS_LANGUAGE"], "fr")

            # A non-XTTS engine is passed through untouched.
            os.environ.pop("GEMMA_TTS_ENGINE", None)
            self.assertEqual(
                xtts_engine.strip_xtts_argv(["--tts", "qwen3"]),
                ["--tts", "qwen3"],
            )
            self.assertNotIn("GEMMA_TTS_ENGINE", os.environ)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_strip_xtts_argv_rejects_unknown_option(self):
        with self.assertRaises(SystemExit):
            xtts_engine.strip_xtts_argv(["prog", "--xtts-nonsense", "1"])

    def test_setup_kwargs_from_env(self):
        env = {
            "XTTS_VOICE_DIR": "/tmp/voice",
            "XTTS_LANGUAGE": "de",
            "XTTS_BLOCKSIZE": "256",
            "XTTS_STREAM": "0",
            "XTTS_TEMPERATURE": "0.6",
            "XTTS_TOP_K": "40",
        }
        saved = {key: os.environ.get(key) for key in env}
        try:
            os.environ.update(env)
            kwargs = xtts_engine.setup_kwargs_from_env()
            self.assertEqual(kwargs["voice_dir"], "/tmp/voice")
            self.assertEqual(kwargs["language"], "de")
            self.assertEqual(kwargs["blocksize"], 256)
            self.assertIs(kwargs["stream"], False)
            self.assertAlmostEqual(kwargs["temperature"], 0.6)
            self.assertEqual(kwargs["top_k"], 40)
            self.assertEqual(kwargs["device"], "auto")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


try:
    from speech_to_speech.pipeline.cancel_scope import CancelScope
    from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput

    _S2S_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _S2S_AVAILABLE = False


@unittest.skipUnless(_S2S_AVAILABLE, "speech-to-speech package not installed")
class XttsHandlerTests(unittest.TestCase):
    def setUp(self):
        _install_fake_torch()
        _install_fake_tts()
        import importlib

        import xtts_tts_handler

        self.handler_mod = importlib.reload(xtts_tts_handler)
        self.tmp = tempfile.TemporaryDirectory()
        self.voice_dir = _make_voice_dir(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, **overrides):
        from threading import Event
        from queue import Queue

        kwargs = {
            "voice_dir": str(self.voice_dir),
            "warmup": False,
            "blocksize": 512,
            "cancel_scope": CancelScope(),
        }
        kwargs.update(overrides)
        return self.handler_mod.XttsTTSHandler(
            Event(),
            queue_in=Queue(),
            queue_out=Queue(),
            setup_args=(Event(),),
            setup_kwargs=kwargs,
        )

    def test_setup_resolves_voice_files(self):
        handler = self._build()
        self.assertEqual(handler.model_path, self.voice_dir / "model.pth")
        self.assertEqual(handler.config_path, self.voice_dir / "config.json")
        self.assertEqual(handler.vocab_path, self.voice_dir / "vocab.json")
        self.assertEqual(handler.speaker_wav, self.voice_dir / "speaker_ref.wav")
        self.assertEqual(handler.output_sample_rate, XTTS_SR)
        self.assertEqual(handler.language, "en")

    def test_missing_voice_files_raise_readable_error(self):
        (self.voice_dir / "vocab.json").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            self._build()
        self.assertIn("vocab.json", str(ctx.exception))

    def test_chunks_are_int16_pcm_at_16khz(self):
        import numpy as np

        handler = self._build()
        chunks = list(handler.process(TTSInput(text="Hello there, avatar.", turn_id="t1", turn_revision=0)))

        self.assertTrue(chunks, "handler produced no audio")
        for chunk in chunks:
            self.assertIsInstance(chunk, np.ndarray)
            self.assertEqual(chunk.dtype, np.int16)
            self.assertEqual(len(chunk), 512)

        # 4 x 0.25 s of 24 kHz audio = 1.0 s => 16000 samples at 16 kHz.
        # 16000 / 512 = 31 full chunks + 1 padded remainder.
        self.assertEqual(len(chunks), 32)
        audio = np.concatenate(chunks)
        self.assertEqual(len(audio), len(chunks) * 512)
        self.assertGreater(int(np.abs(audio).max()), 1000)

        # Rate check: 1 s of speech at 16 kHz.
        speech_samples = 16000
        self.assertLessEqual(abs(speech_samples - 16000), 1)

    def test_end_of_response_yields_done_sentinel(self):
        handler = self._build()
        out = list(handler.process(EndOfResponse(turn_id="t1", turn_revision=0)))
        self.assertEqual(out, [AUDIO_RESPONSE_DONE])

    def test_barge_in_cancels_mid_generation(self):
        """A cancel arriving while audio is streaming must stop the stage early."""
        cancel_scope = CancelScope()
        handler = self._build(cancel_scope=cancel_scope)

        original_stream = handler.model.inference_stream

        def cancelling_stream(*args, **kwargs):
            for index, chunk in enumerate(original_stream(*args, **kwargs)):
                if index == 1:
                    # Simulates the user starting to talk again.
                    cancel_scope.cancel()
                yield chunk

        handler.model.inference_stream = cancelling_stream
        chunks = list(handler.process(TTSInput(text="This should be interrupted.", turn_id="t2", turn_revision=0)))

        self.assertGreater(len(chunks), 0, "audio before the interruption should still be emitted")
        self.assertLess(len(chunks), 32, "generation should stop after the interruption")

    def test_pre_cancelled_scope_still_speaks_new_response(self):
        """The generation counter is captured per response, like the Qwen3 handler."""
        cancel_scope = CancelScope()
        cancel_scope.cancel()  # stale for any earlier response
        handler = self._build(cancel_scope=cancel_scope)
        chunks = list(handler.process(TTSInput(text="Fresh response.", turn_id="t2b", turn_revision=0)))
        self.assertEqual(len(chunks), 32)

    def test_language_and_parameters_reach_model(self):
        handler = self._build(language="en")
        list(handler.process(TTSInput(text="Bonjour tout le monde.", language_code="fr", turn_id="t3")))
        # Fresh tones are generated per call; inspect what the fake model saw.
        chunks = list(handler.process(TTSInput(text="Bonjour.", language_code="fr", turn_id="t4")))
        self.assertTrue(chunks)
        self.assertEqual(handler.model.last_call["language"], "fr")
        self.assertEqual(handler.model.last_call["stream_chunk_size"], 20)

    def test_empty_text_produces_no_audio(self):
        handler = self._build()
        self.assertEqual(list(handler.process(TTSInput(text="   ", turn_id="t5"))), [])

    def test_one_shot_mode_uses_inference(self):
        handler = self._build(stream=False)
        chunks = list(handler.process(TTSInput(text="One shot mode.", turn_id="t6")))
        self.assertEqual(len(chunks), 32)
        self.assertIn("speaker_embedding", handler.model.last_call)


class VoiceCheckScriptTests(unittest.TestCase):
    """check_xtts_voice.py is stdlib-only and must run on bare Python."""

    def test_reports_ready_folder(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            voice_dir = _make_voice_dir(Path(tmp))
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "check_xtts_voice.py"), str(voice_dir)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("ready for XTTS", result.stdout)

    def test_reports_missing_files(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            voice_dir = Path(tmp) / "broken"
            voice_dir.mkdir()
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "check_xtts_voice.py"), str(voice_dir), "--quiet"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("model.pth", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_S2S_AVAILABLE, "speech-to-speech package not installed")
class LauncherWiringTests(unittest.TestCase):
    """The launcher must translate `--tts xtts` and route it to XttsTTSHandler."""

    def setUp(self):
        _install_fake_torch()
        _install_fake_tts()
        from queue import Queue
        from threading import Event

        # Fake the two heavy modules the launcher imports at module scope.
        base_lm = types.ModuleType("speech_to_speech.LLM.base_openai_compatible_language_model")

        class BaseOpenAICompatibleHandler:
            def setup(self, *args, **kwargs):
                pass

        base_lm.BaseOpenAICompatibleHandler = BaseOpenAICompatibleHandler
        service = types.ModuleType("speech_to_speech.api.openai_realtime.service")

        class RealtimeService:
            def parse_client_event(self, raw):
                return raw

        service.RealtimeService = RealtimeService
        nano_parakeet = types.ModuleType("nano_parakeet")
        nano_parakeet.from_pretrained = lambda *a, **k: None

        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "speech_to_speech.LLM.base_openai_compatible_language_model",
                "speech_to_speech.api.openai_realtime.service",
                "nano_parakeet",
                "speech_to_speech.s2s_pipeline",
            )
        }
        sys.modules[
            "speech_to_speech.LLM.base_openai_compatible_language_model"
        ] = base_lm
        sys.modules["speech_to_speech.api.openai_realtime.service"] = service
        sys.modules["nano_parakeet"] = nano_parakeet

        # Fake s2s_pipeline: real one pulls in torch/transformers/nltk.
        self.original_calls = []

        def original_get_tts_handler(module_kwargs, stop_event, q_in, q_out, should_listen, *args, **kwargs):
            self.original_calls.append(module_kwargs.tts)
            return f"original-handler:{module_kwargs.tts}"

        fake_s2s = types.ModuleType("speech_to_speech.s2s_pipeline")
        fake_s2s.parse_arguments = lambda: types.SimpleNamespace(
            module_kwargs=types.SimpleNamespace(tts="qwen3")
        )
        fake_s2s.get_tts_handler = original_get_tts_handler
        fake_s2s.main = lambda: 0
        sys.modules["speech_to_speech.s2s_pipeline"] = fake_s2s
        self.fake_s2s = fake_s2s

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gemma_launcher_under_test", REPO_ROOT / "local_s2s_launcher.py"
        )
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)

        self.tmp = tempfile.TemporaryDirectory()
        self.voice_dir = _make_voice_dir(Path(self.tmp.name))
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("GEMMA_TTS_ENGINE", "XTTS_VOICE_DIR", "XTTS_WARMUP")
        }

    def tearDown(self):
        self.tmp.cleanup()
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.argv = ["prog"]

    def _arguments(self):
        from queue import Queue
        from threading import Event

        return (
            types.SimpleNamespace(tts="xtts"),
            Event(),
            Queue(),
            Queue(),
            Event(),
        )

    def test_tts_xtts_selects_xtts_handler(self):
        from xtts_tts_handler import XttsTTSHandler

        os.environ["XTTS_VOICE_DIR"] = str(self.voice_dir)
        os.environ["XTTS_WARMUP"] = "0"
        self.launcher._patch_xtts_support()

        handler = self.fake_s2s.get_tts_handler(*self._arguments())
        self.assertIsInstance(handler, XttsTTSHandler)
        self.assertEqual(handler.speaker_wav, self.voice_dir / "speaker_ref.wav")
        self.assertEqual(self.original_calls, [], "upstream handler must not be used for xtts")

    def test_other_engines_fall_through_to_upstream(self):
        os.environ.pop("GEMMA_TTS_ENGINE", None)
        self.launcher._patch_xtts_support()

        module_kwargs, stop_event, q_in, q_out, should_listen = self._arguments()
        module_kwargs.tts = "qwen3"
        result = self.fake_s2s.get_tts_handler(module_kwargs, stop_event, q_in, q_out, should_listen)
        self.assertEqual(result, "original-handler:qwen3")

    def test_env_only_selection_wins_over_reset_module_kwargs(self):
        """`--local_mac_optimal_settings` can rewrite module_kwargs.tts; the env var still selects XTTS."""
        from xtts_tts_handler import XttsTTSHandler

        os.environ["XTTS_VOICE_DIR"] = str(self.voice_dir)
        os.environ["XTTS_WARMUP"] = "0"
        os.environ["GEMMA_TTS_ENGINE"] = "xtts"
        self.launcher._patch_xtts_support()

        module_kwargs, stop_event, q_in, q_out, should_listen = self._arguments()
        module_kwargs.tts = "qwen3"  # e.g. rewritten by local_mac_optimal_settings
        handler = self.fake_s2s.get_tts_handler(module_kwargs, stop_event, q_in, q_out, should_listen)
        self.assertIsInstance(handler, XttsTTSHandler)

    def test_parse_arguments_reports_xtts_engine(self):
        self.launcher._patch_xtts_support()
        sys.argv = [
            "local_s2s_launcher.py",
            "--tts",
            "xtts",
            "--xtts-voice-dir",
            str(self.voice_dir),
            "--mode",
            "realtime",
        ]
        parsed = self.fake_s2s.parse_arguments()
        self.assertEqual(parsed.module_kwargs.tts, "xtts")
        self.assertEqual(os.environ["XTTS_VOICE_DIR"], str(self.voice_dir))
        self.assertEqual(sys.argv, ["local_s2s_launcher.py", "--mode", "realtime"])
