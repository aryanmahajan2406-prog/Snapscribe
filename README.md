# SnapScribe — On-Device Ambient Meeting Copilot for Snapdragon PCs

**Built for the Snapdragon® AI Lab Build & Present Challenge (Qualcomm × HP)**

SnapScribe listens to your meetings and work sessions, transcribes them live, tells
speakers apart, and produces rolling summaries — entirely **on-device**, using the
Hexagon NPU on Snapdragon X / X2 Elite HP PCs. No audio, transcript, or summary ever
leaves the machine.

## Why this matters on Snapdragon specifically

Continuous ASR + summarization is a sustained, moderate-compute workload — exactly
where NPU efficiency (TOPS/watt) beats running the same models on CPU/GPU. Running
this all day should cost you a sliver of battery, not your afternoon.

| Component        | Model                              | Runs on         |
|-------------------|-------------------------------------|------------------|
| VAD               | WebRTC VAD                          | CPU (negligible) |
| Speech-to-text    | Whisper (base/small), QNN-compiled  | Hexagon NPU      |
| Summarization     | Phi-3.5-mini-instruct, QNN ONNX     | Hexagon NPU      |
| Storage / search  | SQLite + FTS5                       | CPU              |
| UI                | Tray icon + live caption overlay    | CPU              |

## Architecture

```
 Mic + System Audio (WASAPI loopback)
            │
            ▼
     ┌─────────────┐
     │     VAD      │  chunks speech, drops silence
     └──────┬───────┘
            ▼
     ┌─────────────┐        QNNExecutionProvider
     │  Whisper ASR │◄────── (Hexagon NPU, via ONNX Runtime)
     └──────┬───────┘
            ▼
     ┌─────────────┐
     │  Diarization │  mic-channel vs loopback-channel tagging
     └──────┬───────┘
            ▼
     ┌─────────────┐
     │  Transcript  │──► SQLite (FTS5 full-text search)
     │    Store     │
     └──────┬───────┘
            │ every N minutes
            ▼
     ┌─────────────┐        QNNExecutionProvider / onnxruntime-genai
     │ Summarizer   │◄────── (Hexagon NPU)
     │ (Phi-3.5)    │
     └──────┬───────┘
            ▼
     ┌─────────────┐
     │  Overlay UI  │  live captions + rolling summary + tray icon
     └─────────────┘
```

## Hardware target

- HP OmniBook (or equivalent) with Snapdragon X Elite (45 TOPS) or X2 Elite/X2 Plus
  (80 TOPS) Hexagon NPU.
- Windows 11, Copilot+ PC certified.

## Running without the physical device (dev mode)

Every module reads its execution provider from `config/settings.yaml`. During
development (before hardware access was available), everything runs against
`CPUExecutionProvider` — same code path, same models, just slower. Model
compilation and **real NPU performance validation** were done remotely against a
cloud-hosted **Snapdragon X Elite CRD** via **Qualcomm AI Hub** — see
`scripts/export_models.py` and `docs/ai_hub_validation.md`.

To switch to real hardware once available, set in `config/settings.yaml`:
```yaml
execution_provider: QNNExecutionProvider
qnn_backend_path: QnnHtp.dll
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# one-time: compile + profile models via Qualcomm AI Hub
python scripts/export_models.py

# run
python -m src.main
```

## Project layout

```
src/
  audio_capture.py   # WASAPI loopback + mic capture
  vad.py              # speech segmentation
  transcribe.py       # Whisper ONNX inference (QNN/CPU)
  diarize.py           # speaker tagging
  summarize.py          # Phi-3.5 rolling summarization (QNN/CPU)
  store.py              # SQLite + FTS5 transcript store
  ui_overlay.py          # tray icon + live caption overlay
  main.py                 # orchestrator
scripts/
  export_models.py          # Qualcomm AI Hub compile/profile jobs
docs/
  ai_hub_validation.md        # NPU perf numbers from AI Hub
  privacy.md                   # what stays on-device, what doesn't (nothing)
config/
  settings.yaml
```

## Status

Prototype built for the Build & Present Challenge submission. Core pipeline is
functional against CPU fallback; NPU (QNN) path validated via Qualcomm AI Hub
cloud-hosted Snapdragon X Elite CRD pending final on-device confirmation.

## License

MIT
