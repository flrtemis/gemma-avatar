#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler


_ORIGINAL_SETUP = BaseOpenAICompatibleHandler.setup


def _patched_setup(self, *args, **kwargs):
    kwargs.setdefault("request_timeout_s", float(os.environ.get("LOCAL_LLM_REQUEST_TIMEOUT_S", "300")))
    return _ORIGINAL_SETUP(self, *args, **kwargs)


BaseOpenAICompatibleHandler.setup = _patched_setup


def _patch_qwen3_gguf_paths() -> None:
    talker_path = os.environ.get("QWEN3_GGUF_TALKER_PATH", "").strip()
    codec_path = os.environ.get("QWEN3_GGUF_CODEC_PATH", "").strip()
    if not talker_path or not codec_path:
        return

    talker = Path(talker_path)
    codec = Path(codec_path)
    if not talker.is_file() or not codec.is_file():
        return

    from qwentts_cpp import models as qwentts_models

    def _resolve_local_gguf_paths(model_id: str, **_kwargs):
        return talker, codec

    qwentts_models.resolve_gguf_paths = _resolve_local_gguf_paths


_patch_qwen3_gguf_paths()


from speech_to_speech.s2s_pipeline import main


if __name__ == "__main__":
    sys.exit(main())