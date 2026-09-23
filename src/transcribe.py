"""
Speech-to-text via Whisper, running on the Hexagon NPU through ONNX Runtime's
QNN execution provider (falls back to CPU EP during development).

Model source: compiled/profiled via Qualcomm AI Hub
(qai_hub_models.models.whisper_base_en) — see scripts/export_models.py.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio_capture import pcm_bytes_to_float32


class WhisperTranscriber:
    def __init__(
        self,
        model_dir: str,
        execution_provider: str = "CPUExecutionProvider",
        qnn_backend_path: str = "QnnHtp.dll",
    ):
        self.model_dir = Path(model_dir)
        self._session = None
        self._execution_provider = execution_provider
        self._qnn_backend_path = qnn_backend_path
        self._app = None  # lazily-loaded qai_hub_models WhisperApp, if available

    def load(self):
        """
        Load the ASR model. Two code paths:

        1. Preferred: qai_hub_models ships a WhisperApp wrapper per exported
           model that handles mel-spectrogram preprocessing, encoder/decoder
           looping and BPE decoding for you.
        2. Fallback: raw onnxruntime.InferenceSession on an encoder/decoder
           ONNX pair, for environments where qai_hub_models isn't installed
           (e.g. quick CPU-only smoke testing).
        """
        try:
            from qai_hub_models.models.whisper_base_en.app import WhisperApp
            from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn

            provider_options = None
            providers = [self._execution_provider]
            if self._execution_provider == "QNNExecutionProvider":
                provider_options = [{"backend_path": self._qnn_backend_path}]

            model = WhisperBaseEn.from_pretrained()
            self._app = WhisperApp(
                model,
                providers=providers,
                provider_options=provider_options,
            )
        except ImportError:
            # Raw ONNX Runtime fallback — expects encoder.onnx / decoder.onnx
            # already exported into model_dir.
            providers = [self._execution_provider]
            provider_options = None
            if self._execution_provider == "QNNExecutionProvider":
                provider_options = [{"backend_path": self._qnn_backend_path}]

            self._encoder = ort.InferenceSession(
                str(self.model_dir / "encoder.onnx"),
                providers=providers,
                provider_options=provider_options,
            )
            self._decoder = ort.InferenceSession(
                str(self.model_dir / "decoder.onnx"),
                providers=providers,
                provider_options=provider_options,
            )

    def transcribe_pcm(self, pcm_bytes: bytes) -> str:
        """Transcribe one speech segment (raw int16 PCM) to text."""
        audio = pcm_bytes_to_float32(pcm_bytes)

        if self._app is not None:
            return self._app.transcribe(audio)

        # Minimal raw-ONNX path: real deployments should use the qai_hub_models
        # WhisperApp above, which handles mel-spectrogram + beam search decoding.
        # This fallback is left intentionally explicit rather than silently
        # wrong — wire up your own preprocessing here if you're not using
        # qai_hub_models.
        raise NotImplementedError(
            "Raw ONNX Whisper path needs mel-spectrogram preprocessing + "
            "decoder loop wired up. Install qai_hub_models for the "
            "batteries-included WhisperApp path used in development."
        )
