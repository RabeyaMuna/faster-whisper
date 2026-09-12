from faster_whisper.audio import decode_audio
from faster_whisper.version import __version__


def __getattr__(name):
    if name in ("BatchedInferencePipeline", "WhisperModel"):
        from faster_whisper.transcribe import BatchedInferencePipeline, WhisperModel

        return (
            BatchedInferencePipeline
            if name == "BatchedInferencePipeline"
            else WhisperModel
        )
    if name in ("available_models", "download_model", "format_timestamp"):
        from faster_whisper.utils import (
            available_models,
            download_model,
            format_timestamp,
        )

        return {
            "available_models": available_models,
            "download_model": download_model,
            "format_timestamp": format_timestamp,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]
