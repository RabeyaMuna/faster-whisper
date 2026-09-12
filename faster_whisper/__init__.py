from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import (
    AsyncBatchedInferencePipeline,
    BatchedInferencePipeline,
    WhisperModel,
)

try:
    from faster_whisper.utils import available_models, download_model, format_timestamp
except ImportError:
    # 'requests' (used by faster_whisper.utils) may be missing in some environments.
    # Defer the import error until these helpers are actually used.
    available_models = download_model = format_timestamp = None
from faster_whisper.version import __version__

__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "AsyncBatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]
