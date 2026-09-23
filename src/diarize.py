"""
Speaker tagging.

v1 (this file): cheap and reliable — tag by capture channel. Audio from the
mic is "You", audio from WASAPI loopback is "Them". This covers the most
common case (you + one remote call) without any extra model or NPU cost.

v2 (future work): embedding-based clustering (e.g. ECAPA-TDNN) to separate
multiple remote speakers within the loopback channel itself. Left out of
this submission's scope to keep the NPU budget on ASR + summarization,
which matter more for the demo.
"""

from dataclasses import dataclass


@dataclass
class TaggedSegment:
    speaker: str
    text: str
    timestamp: float


def tag_speaker(channel: str) -> str:
    return "You" if channel == "mic" else "Them"
