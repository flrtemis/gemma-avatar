"""Configuration plumbing for the optional XTTS-v2 (Coqui) TTS engine.

This module is deliberately dependency-free (standard library only), so it can
be imported by the launcher before Torch/Coqui are touched, and unit-tested on
a bare Python install.

Responsibilities
----------------
* engine aliases (``xtts``, ``xtts-v2``, ``coqui-xtts``, ...)
* translating ``--tts xtts`` / ``--xtts-*`` CLI options into ``XTTS_*`` env vars
* reading those env vars into :meth:`XttsTTSHandler.setup` kwargs
* small text/language helpers shared with the handler
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_CHARS = 400
DEFAULT_BLOCKSIZE = 512

XTTS_TTS_ALIASES = frozenset({"xtts", "xtts-v2", "xtts_v2", "coqui-xtts", "coqui_xtts"})

# ``--xtts-<key>`` -> environment variable consumed by setup_kwargs_from_env().
XTTS_CLI_TO_ENV = {
    "voice_dir": "XTTS_VOICE_DIR",
    "model_path": "XTTS_MODEL_PATH",
    "config_path": "XTTS_CONFIG_PATH",
    "vocab_path": "XTTS_VOCAB_PATH",
    "speaker_wav": "XTTS_SPEAKER_WAV",
    "language": "XTTS_LANGUAGE",
    "device": "XTTS_DEVICE",
    "dtype": "XTTS_DTYPE",
    "speed": "XTTS_SPEED",
    "blocksize": "XTTS_BLOCKSIZE",
    "stream": "XTTS_STREAM",
    "stream_chunk_size": "XTTS_STREAM_CHUNK_SIZE",
    "overlap_wav_len": "XTTS_OVERLAP_WAV_LEN",
    "temperature": "XTTS_TEMPERATURE",
    "length_penalty": "XTTS_LENGTH_PENALTY",
    "repetition_penalty": "XTTS_REPETITION_PENALTY",
    "top_k": "XTTS_TOP_K",
    "top_p": "XTTS_TOP_P",
    "do_sample": "XTTS_DO_SAMPLE",
    "gpt_cond_len": "XTTS_GPT_COND_LEN",
    "max_ref_len": "XTTS_MAX_REF_LEN",
    "sound_norm_refs": "XTTS_SOUND_NORM_REFS",
    "max_chars": "XTTS_MAX_CHARS",
    "warmup": "XTTS_WARMUP",
}

# Reference clip filenames that voice-export tools commonly use.
VOICE_REF_CANDIDATES = (
    "speaker_ref.wav",
    "speaker_reference.wav",
    "reference.wav",
    "speaker.wav",
    "sample.wav",
    "ref.wav",
)
VOICE_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

# XTTS-v2 language codes (coqui spells Chinese "zh-cn").
XTTS_LANGUAGES = frozenset(
    {
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
        "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    }
)

# Whisper / langdetect style codes -> XTTS codes.
LANGUAGE_CODE_MAP = {
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
    "es-es": "es",
    "es-mx": "es",
    "fr-fr": "fr",
    "de-de": "de",
    "it-it": "it",
    "pt-br": "pt",
    "pt-pt": "pt",
    "pl-pl": "pl",
    "tr-tr": "tr",
    "ru-ru": "ru",
    "nl-nl": "nl",
    "cs-cz": "cs",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-hans": "zh-cn",
    "zh-hant": "zh-cn",
    "zh-tw": "zh-cn",
    "cmn": "zh-cn",
    "chinese": "zh-cn",
    "ja-jp": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko-kr": "ko",
    "kr": "ko",
    "korean": "ko",
    "hu-hu": "hu",
    "hi-in": "hi",
}

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y"})

_MARKDOWN_NOISE = re.compile(r"[*_`#>|~\[\]{}]")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_SPEAKABLE = re.compile(r"[0-9A-Za-zÀ-ÿ\u0080-\uffff]")


def normalize_engine(value: Optional[str]) -> Optional[str]:
    """Return ``"xtts"`` for any XTTS alias, else the lower-cased input."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in XTTS_TTS_ALIASES:
        return "xtts"
    return normalized


def resolve_language(code: Optional[str], default: str = DEFAULT_LANGUAGE) -> str:
    """Map a pipeline language code onto a valid XTTS-v2 language code."""
    fallback = (default or DEFAULT_LANGUAGE).strip().lower()
    if fallback not in XTTS_LANGUAGES:
        fallback = fallback.split("-")[0]
    if fallback not in XTTS_LANGUAGES:
        fallback = DEFAULT_LANGUAGE

    if not code:
        return fallback

    normalized = str(code).strip().lower()
    if normalized in XTTS_LANGUAGES:
        return normalized
    if normalized in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[normalized]
    base = normalized.split("-")[0]
    if base in XTTS_LANGUAGES:
        return base
    return fallback


def clean_text(text: str) -> str:
    """Strip markdown noise so XTTS does not read symbols aloud."""
    if not text:
        return ""
    cleaned = _WHITESPACE.sub(" ", _MARKDOWN_NOISE.sub("", text)).strip()
    if not _SPEAKABLE.search(cleaned):
        return ""
    return cleaned


def split_for_xtts(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split long text on sentence boundaries (XTTS caps text tokens per call)."""
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer.strip())
            buffer = sentence
    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def _env_text(name: str) -> Optional[str]:
    value = (os.environ.get(name) or "").strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_text(name)
    if raw is None:
        return default
    return raw.lower() in _TRUE_VALUES


def _env_number(name: str, default: float, cast):
    raw = _env_text(name)
    if raw is None:
        return default
    try:
        return cast(float(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r (expected a number)", name, raw)
        return default


def setup_kwargs_from_env() -> dict:
    """Build ``XttsTTSHandler.setup`` kwargs from ``XTTS_*`` environment vars."""
    return {
        "voice_dir": _env_text("XTTS_VOICE_DIR"),
        "model_path": _env_text("XTTS_MODEL_PATH"),
        "config_path": _env_text("XTTS_CONFIG_PATH"),
        "vocab_path": _env_text("XTTS_VOCAB_PATH"),
        "speaker_wav": _env_text("XTTS_SPEAKER_WAV"),
        "language": _env_text("XTTS_LANGUAGE") or DEFAULT_LANGUAGE,
        "device": _env_text("XTTS_DEVICE") or "auto",
        "dtype": _env_text("XTTS_DTYPE") or "fp32",
        "speed": _env_number("XTTS_SPEED", 1.0, float),
        "blocksize": _env_number("XTTS_BLOCKSIZE", DEFAULT_BLOCKSIZE, int),
        "stream": _env_bool("XTTS_STREAM", True),
        "stream_chunk_size": _env_number("XTTS_STREAM_CHUNK_SIZE", 20, int),
        "overlap_wav_len": _env_number("XTTS_OVERLAP_WAV_LEN", 1024, int),
        "temperature": _env_number("XTTS_TEMPERATURE", 0.75, float),
        "length_penalty": _env_number("XTTS_LENGTH_PENALTY", 1.0, float),
        "repetition_penalty": _env_number("XTTS_REPETITION_PENALTY", 10.0, float),
        "top_k": _env_number("XTTS_TOP_K", 50, int),
        "top_p": _env_number("XTTS_TOP_P", 0.85, float),
        "do_sample": _env_bool("XTTS_DO_SAMPLE", True),
        "gpt_cond_len": _env_number("XTTS_GPT_COND_LEN", 6, int),
        "max_ref_len": _env_number("XTTS_MAX_REF_LEN", 30, int),
        "sound_norm_refs": _env_bool("XTTS_SOUND_NORM_REFS", False),
        "max_chars": _env_number("XTTS_MAX_CHARS", DEFAULT_MAX_CHARS, int),
        "warmup": _env_bool("XTTS_WARMUP", True),
    }


def strip_xtts_argv(argv: list[str]) -> list[str]:
    """Remove XTTS-only CLI options from ``argv``, recording them as env vars.

    The upstream argument parser validates ``--tts`` against a fixed literal
    list that does not include XTTS, and knows nothing about ``--xtts-*``
    options, so they are translated here instead of being passed through.
    Setting ``GEMMA_TTS_ENGINE=xtts`` (or any ``--xtts-*`` option) selects the
    XTTS handler.
    """
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]

        if arg == "--tts" and index + 1 < len(argv) and normalize_engine(argv[index + 1]) == "xtts":
            os.environ["GEMMA_TTS_ENGINE"] = "xtts"
            index += 2
            continue
        if arg.startswith("--tts=") and normalize_engine(arg.split("=", 1)[1]) == "xtts":
            os.environ["GEMMA_TTS_ENGINE"] = "xtts"
            index += 1
            continue

        if arg.startswith("--xtts-") or arg.startswith("--xtts_"):
            option, _, inline_value = arg[2:].partition("=")
            key = option.replace("-", "_")[len("xtts_"):]
            env_name = XTTS_CLI_TO_ENV.get(key)
            if env_name is None:
                known = ", ".join(f"--xtts-{name.replace('_', '-')}" for name in XTTS_CLI_TO_ENV)
                raise SystemExit(f"Unknown XTTS option {arg}. Known options: {known}")
            if inline_value:
                os.environ[env_name] = inline_value
                index += 1
            else:
                if index + 1 >= len(argv):
                    raise SystemExit(f"Option {arg} expects a value")
                os.environ[env_name] = argv[index + 1]
                index += 2
            os.environ["GEMMA_TTS_ENGINE"] = "xtts"
            continue

        remaining.append(arg)
        index += 1

    # `--tts xtts` sets the engine; also allow GEMMA_TTS_ENGINE alone.
    return remaining
