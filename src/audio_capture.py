"""
Audio capture: microphone + system-audio (WASAPI loopback) on Windows.

Produces raw 16kHz mono PCM frames on two logical channels:
  - "mic"      -> what you say
  - "loopback" -> what everyone else in the call says (system output)

This channel split is also what src/diarize.py uses for cheap v1 speaker
tagging, without needing a full diarization model.
"""

import math
import queue
import threading

import numpy as np
from scipy.signal import resample_poly

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
        device_info = self._pa.get_device_info_by_index(device_index)
        native_rate = int(device_info.get("defaultSampleRate", SAMPLE_RATE))
        is_loopback = device_info.get("isLoopbackDevice", False)

        if is_loopback:
            channels = int(device_info.get("maxInputChannels", 0))
            if channels == 0:
                channels = int(device_info.get("maxOutputChannels", 2))
            rate = native_rate
        else:
            try:
                test_stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                )
                test_stream.close()
                channels = 1
                rate = SAMPLE_RATE
            except Exception:
                channels = max(1, int(device_info.get("maxInputChannels", 1)))
                rate = native_rate

        frames_per_buffer = int(rate * 0.03)  # 30ms buffer
        remainder_pcm = b""

        def _callback(in_data, frame_count, time_info, status):
            nonlocal remainder_pcm
            data = np.frombuffer(in_data, dtype=np.int16)
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)

            if rate != SAMPLE_RATE:
                gcd = math.gcd(SAMPLE_RATE, rate)
                data = resample_poly(
                    data.astype(np.float32), SAMPLE_RATE // gcd, rate // gcd
                ).astype(np.int16)

            pcm_out = remainder_pcm + data.tobytes()
            target_bytes = CHUNK_SAMPLES * 2  # 480 samples * 2 bytes = 960 bytes

            offset = 0
            while offset + target_bytes <= len(pcm_out):
                chunk = pcm_out[offset : offset + target_bytes]
                self._frames.put((channel_label, chunk))
                offset += target_bytes
            remainder_pcm = pcm_out[offset:]

            return (None, pyaudio.paContinue)

        return self._pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
            stream_callback=_callback,
        )

    def start(self):
        if self.capture_mic:
            try:
                mic_info = self._pa.get_default_input_device_info()
                self._streams.append(self._make_stream(mic_info["index"], "mic"))
            except Exception as e:
                print(f"[AudioCapture] Warning: could not initialize microphone: {e}")

        if self.capture_system_audio:
            try:
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
            except Exception as e:
                print(f"[AudioCapture] Warning: could not initialize WASAPI loopback: {e}")

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
