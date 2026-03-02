"""
tts_stt_benchmark – automated TTS/STT benchmark harness.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tts-stt-benchmark")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
