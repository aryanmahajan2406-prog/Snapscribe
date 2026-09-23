"""
SnapScribe orchestrator.

Wires together: audio capture -> VAD segmentation -> Whisper transcription
(NPU) -> speaker tagging -> storage -> periodic summarization (NPU) -> UI.

Run with: python -m src.main
"""

import threading
import time
import uuid

import yaml

from .audio_capture import AudioCapture
from .diarize import tag_speaker
from .store import TranscriptStore
from .summarize import RollingSummarizer
from .transcribe import WhisperTranscriber
from .ui_overlay import run_ui
from .vad import SpeechSegmenter


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SnapScribeApp:
    def __init__(self, config: dict):
        self.config = config
        self.session_id = str(uuid.uuid4())
        self._stop_event = threading.Event()

        self.store = TranscriptStore(config["storage"]["db_path"])

        self.transcriber = WhisperTranscriber(
            model_dir=config["asr"]["model_dir"],
            execution_provider=config["execution_provider"],
            qnn_backend_path=config["qnn_backend_path"],
        )

        self.summarizer = RollingSummarizer(
            model_dir=config["summarizer"]["model_dir"],
            max_tokens=config["summarizer"]["max_summary_tokens"],
        )

        self.segmenter = SpeechSegmenter(
            silence_timeout_ms=config["audio"]["silence_timeout_ms"]
        )

        self.capture = AudioCapture(
            capture_mic=config["audio"]["capture_mic"],
            capture_system_audio=config["audio"]["capture_system_audio"],
        )

        self.overlay = None

    def start(self):
        print(f"[SnapScribe] Loading models (provider={self.config['execution_provider']})...")
        self.transcriber.load()
        self.summarizer.load()

        print("[SnapScribe] Starting UI...")
        self.overlay = run_ui(
            overlay_enabled=self.config["ui"]["show_live_captions"],
            on_quit=self.stop,
        )

        print("[SnapScribe] Starting audio capture...")
        self.capture.start()

        threading.Thread(target=self._transcription_loop, daemon=True).start()
        threading.Thread(target=self._summarization_loop, daemon=True).start()

        print("[SnapScribe] Running. Ctrl+C to stop.")
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def _transcription_loop(self):
        for channel, pcm_frame in self.capture.frames():
            result = self.segmenter.push_frame(channel, pcm_frame)
            if result is None:
                continue
            seg_channel, pcm_segment = result
            text = self.transcriber.transcribe_pcm(pcm_segment)
            if not text.strip():
                continue
            speaker = tag_speaker(seg_channel)
            self.store.add_segment(self.session_id, speaker, text)
            if self.overlay is not None:
                self.overlay.push_line(speaker, text)
            print(f"[{speaker}] {text}")

    def _summarization_loop(self):
        interval = self.config["summarizer"]["interval_minutes"] * 60
        while not self._stop_event.is_set():
            time.sleep(interval)
            transcript = self.store.get_transcript(self.session_id)
            if not transcript.strip():
                continue
            summary = self.summarizer.summarize(transcript)
            self.store.add_summary(self.session_id, summary)
            print(f"\n[Summary]\n{summary}\n")

    def stop(self):
        self._stop_event.set()
        self.capture.stop()
        self.store.close()


def main():
    config = load_config()
    app = SnapScribeApp(config)
    app.start()


if __name__ == "__main__":
    main()
