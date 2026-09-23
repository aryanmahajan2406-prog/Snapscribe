"""
Voice activity detection and segmentation.

Groups raw 30ms audio frames into speech segments, dropping silence, so the
ASR model (running on the NPU) is only invoked on actual speech instead of
transcribing dead air all day.
"""

import collections

import webrtcvad

from .audio_capture import SAMPLE_RATE


class SpeechSegmenter:
    def __init__(self, aggressiveness: int = 2, silence_timeout_ms: int = 800):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_timeout_frames = silence_timeout_ms // 30  # 30ms frames
        self._buffers: dict[str, list[bytes]] = collections.defaultdict(list)
        self._silence_counts: dict[str, int] = collections.defaultdict(int)

    def push_frame(self, channel: str, pcm_frame: bytes):
        """
        Feed one 30ms PCM frame for a channel ("mic"/"loopback").
        Returns a completed segment (channel, pcm_bytes) once speech ends,
        otherwise None.
        """
        is_speech = self._vad.is_speech(pcm_frame, SAMPLE_RATE)

        if is_speech:
            self._buffers[channel].append(pcm_frame)
            self._silence_counts[channel] = 0
            return None

        # silence
        if self._buffers[channel]:
            self._silence_counts[channel] += 1
            if self._silence_counts[channel] >= self._silence_timeout_frames:
                segment = b"".join(self._buffers[channel])
                self._buffers[channel] = []
                self._silence_counts[channel] = 0
                return channel, segment
        return None

    def flush(self, channel: str):
        """Force-close any in-progress segment (e.g. on shutdown)."""
        if self._buffers[channel]:
            segment = b"".join(self._buffers[channel])
            self._buffers[channel] = []
            self._silence_counts[channel] = 0
            return channel, segment
        return None
