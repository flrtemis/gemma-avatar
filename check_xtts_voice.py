#!/usr/bin/env python3
"""Validate a Coqui XTTS-v2 voice folder before starting Gemma Avatar.

Checks the files that ``--tts xtts`` (see TTS-XTTS.md) needs, reads the
checkpoint config, and inspects the reference clip with the standard library
only — so it runs on a fresh Windows/WSL Python without any ML dependencies.

Usage
-----
    python check_xtts_voice.py "C:/Users/l3ung/forge/voices/my-voice"
    python check_xtts_voice.py               # uses $XTTS_VOICE_DIR
    python check_xtts_voice.py <dir> --quiet # only problems (used by the launcher)

Exit code 0 = ready to run, 1 = blocking problem.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import wave
from pathlib import Path

REQUIRED_FILES = ("model.pth", "config.json", "vocab.json")
OPTIONAL_FILES = ("speakers_xtts.pth", "voice.json")
REFERENCE_CANDIDATES = (
    "speaker_ref.wav",
    "speaker_reference.wav",
    "reference.wav",
    "speaker.wav",
    "sample.wav",
    "ref.wav",
)
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 30.0


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"<value>": type(data).__name__}
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        return {"<parse error>": str(exc)}


def _find_reference(voice_dir: Path) -> Path | None:
    for name in REFERENCE_CANDIDATES:
        candidate = voice_dir / name
        if candidate.is_file():
            return candidate
    audio = sorted(p for p in voice_dir.glob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    return audio[0] if audio else None


def _describe_wav(path: Path) -> dict:
    info: dict = {"path": str(path), "size": path.stat().st_size}
    if path.suffix.lower() != ".wav":
        info["note"] = "non-WAV reference; XTTS will decode it with soundfile/librosa"
        return info
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 0
            info.update(
                channels=handle.getnchannels(),
                sample_rate=rate,
                sample_width=handle.getsampwidth(),
                duration_s=(frames / rate) if rate else 0.0,
            )
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def check(voice_dir: Path, quiet: bool = False) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    def say(message: str = "") -> None:
        if not quiet:
            print(message)

    say(f"XTTS voice folder: {voice_dir}")

    if not voice_dir.is_dir():
        print(f"ERROR: folder not found: {voice_dir}", file=sys.stderr)
        return 1

    for name in REQUIRED_FILES:
        path = voice_dir / name
        if path.is_file() and path.stat().st_size > 0:
            say(f"  [ok]   {name:<20} {_human_size(path.stat().st_size)}")
        else:
            problems.append(f"missing or empty: {name}")
            say(f"  [FAIL] {name:<20} missing or empty")

    for name in OPTIONAL_FILES:
        path = voice_dir / name
        if path.is_file():
            say(f"  [ok]   {name:<20} {_human_size(path.stat().st_size)} (optional)")

    config = _load_json(voice_dir / "config.json") if (voice_dir / "config.json").is_file() else {}
    if config:
        model_name = config.get("model", "<unknown>")
        model_args = config.get("model_args", {}) if isinstance(config.get("model_args"), dict) else {}
        languages = config.get("languages") or []
        out_rate = model_args.get("output_sample_rate")
        tokenizer_file = model_args.get("tokenizer_file")

        say(f"  config: model={model_name!r} languages={len(languages)} output_sample_rate={out_rate}")
        if model_name not in ("xtts", None) and model_name != "<unknown>":
            warnings.append(
                f"config.json declares model={model_name!r}; this handler is written for XTTS-v2 "
                "checkpoints (model='xtts')."
            )
        tokenizer_text = str(tokenizer_file or "")
        looks_absolute = tokenizer_text.startswith(("/", "\\\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", tokenizer_text))
        if tokenizer_text and looks_absolute:
            notes.append(
                f"config.json model_args.tokenizer_file points at {tokenizer_file!r}; "
                "that path is ignored because vocab.json in this folder is used instead."
            )
        if out_rate and int(out_rate) != 24000:
            notes.append(
                f"output_sample_rate is {out_rate} Hz; audio will be resampled to the pipeline's 16 kHz."
            )

    voice_meta_path = voice_dir / "voice.json"
    if voice_meta_path.is_file():
        meta = _load_json(voice_meta_path)
        keys = ", ".join(list(meta)[:8]) or "<empty>"
        notes.append(
            f"voice.json keys: {keys}. This file is informational only — XTTS clones from the "
            "reference clip, so nothing in voice.json is required."
        )

    reference = _find_reference(voice_dir)
    if reference is None:
        problems.append("no reference audio (expected speaker_ref.wav or similar)")
        say("  [FAIL] reference audio    missing (need 3-30 s of clean speech)")
    else:
        info = _describe_wav(reference)
        if "duration_s" in info:
            duration = float(info["duration_s"])
            say(
                f"  [ok]   {reference.name:<20} {info['sample_rate']} Hz, "
                f"{info['channels']} ch, {duration:.2f}s"
            )
            if duration < MIN_REF_SECONDS:
                warnings.append(
                    f"reference clip is {duration:.2f}s; XTTS clones best from at least "
                    f"{MIN_REF_SECONDS:.0f}s of speech."
                )
            if duration > MAX_REF_SECONDS:
                notes.append(
                    f"reference clip is {duration:.1f}s; only the first {MAX_REF_SECONDS:.0f}s "
                    "(XTTS_MAX_REF_LEN) is used."
                )
            if info.get("channels", 1) > 1:
                notes.append("reference is not mono; XTTS downmixes it automatically.")
        elif "error" in info:
            warnings.append(f"could not read reference header: {info['error']}")
            say(f"  [ok]   {reference.name:<20} present (header unreadable)")
        else:
            say(f"  [ok]   {reference.name:<20} present ({info.get('note', 'audio')})")

    if not (voice_dir / "speakers_xtts.pth").is_file():
        notes.append(
            "speakers_xtts.pth not present — fine for reference-clip cloning; only needed for "
            "preset speaker names inside the checkpoint."
        )

    if notes and not quiet:
        say()
        say("Notes:")
        for note in notes:
            say(f"  - {note}")

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if problems:
        print()
        print("Problems:")
        for problem in problems:
            print(f"  - {problem}")

    if not quiet:
        print()
        if problems:
            print("Result: NOT ready — fix the problems above.")
        else:
            print("Result: ready for XTTS.")
            print()
            print("Start the avatar stack with:")
            print(f'  TTS_ENGINE=xtts XTTS_VOICE_DIR="{voice_dir}" ./run-all-local.sh')

    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Coqui XTTS-v2 voice folder for Gemma Avatar.")
    parser.add_argument("voice_dir", nargs="?", default=os.environ.get("XTTS_VOICE_DIR") or None)
    parser.add_argument("--quiet", action="store_true", help="Only print problems and warnings.")
    args = parser.parse_args()

    if not args.voice_dir:
        parser.error("no voice folder given and XTTS_VOICE_DIR is not set")
    return check(Path(args.voice_dir).expanduser(), quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
