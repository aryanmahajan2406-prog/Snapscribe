"""
Audio capture: microphone + system-audio (WASAPI loopback) on Windows.

Produces raw 16kHz mono PCM frames on two logical channels:
  - "mic"      -> what you say
  - "loopback" -> what everyone else in the call says (system output)

This channel split is also what src/diarize.py uses for cheap v1 speaker
tagging, without needing a full diarization model.
"""

import queue
import threading

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover - only available on Windows
    pyaudio = None

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 480  # 30ms @ 16kHz, matches webrtcvad frame size


class AudioCapture:
    """Captures mic + loopback audio into a shared queue of (channel, pcm_bytes)."""

    def __init__(self, capture_mic: bool = True, capture_system_audio: bool = True):
        if pyaudio is None:
            raise RuntimeError(
                "pyaudiowpatch is required for audio capture and only runs on "
                "Windows. Install with: pip install pyaudiowpatch"
            )
        self.capture_mic = capture_mic
        self.capture_system_audio = capture_system_audio
        self._pa = pyaudio.PyAudio()
        self._frames: "queue.Queue[tuple[str, bytes]]" = queue.Queue()
        self._streams: list = []
        self._stop_event = threading.Event()

    def _make_stream(self, device_index: int, channel_label: str):
        def _callback(in_data, frame_count, time_info, status):
            self._frames.put((channel_label, in_data))
            return (None, pyaudio.paContinue)

        return self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SAMPLES,
            stream_callback=_callback,
        )

    def start(self):
        if self.capture_mic:
            mic_info = self._pa.get_default_input_device_info()
            self._streams.append(self._make_stream(mic_info["index"], "mic"))

        if self.capture_system_audio:
            # WASAPI loopback device: the default speaker's "loopback" twin,
            # exposed by pyaudiowpatch.
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self._pa.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )
            if not default_speakers.get("isLoopbackDevice", False):
                for loopback in self._pa.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
            self._streams.append(
                self._make_stream(default_speakers["index"], "loopback")
            )

        for s in self._streams:
            s.start_stream()

    def frames(self):
        """Generator yielding (channel_label, pcm_int16_bytes) as they arrive."""
        while not self._stop_event.is_set():
            try:
                yield self._frames.get(timeout=0.5)
            except queue.Empty:
                continue

    def stop(self):
        self._stop_event.set()
        for s in self._streams:
            s.stop_stream()
            s.close()
        self._pa.terminate()


def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to float32 [-1, 1] array for the ASR model."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0
