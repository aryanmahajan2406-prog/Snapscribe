# SnapScribe Architecture & System Design

SnapScribe is an on-device ambient meeting copilot engineered for Snapdragon® X Elite Copilot+ PCs. It captures ambient meeting audio, performs voice activity detection, transcribes speech with Whisper, attributes speakers, indexes transcripts locally with full-text search, generates periodic rolling summaries with Phi-3.5, and renders live captions.

---

## 1. Pipeline Stages & Data Flow

The data flow strictly follows the modular structure in `src/`:

```mermaid
flowchart TD
    subgraph AudioCapture["1. Audio Capture (src/audio_capture.py)"]
        Mic["Microphone Stream"] --> Q["queue.Queue (16kHz mono PCM)"]
        Loopback["WASAPI Loopback Stream"] --> Q
    end

    subgraph VAD["2. Voice Activity Detection (src/vad.py)"]
        Q --> Frame["30ms PCM Frames (480 samples)"]
        Frame --> Segmenter["SpeechSegmenter (webrtcvad)"]
        Segmenter -->|Silence Timeout (800ms)| Segment["Contiguous Speech Segment (channel, bytes)"]
    end

    subgraph ASR["3. Speech-to-Text (src/transcribe.py)"]
        Segment --> Norm["pcm_bytes_to_float32"]
        Norm --> Whisper["WhisperTranscriber (whisper_base)"]
        Whisper --> Text["Transcribed Text"]
    end

    subgraph Diarization["4. Speaker Tagging (src/diarize.py)"]
        Text --> Tag["tag_speaker: 'mic' -> 'You', 'loopback' -> 'Them'"]
        Tag --> TaggedSeg["TaggedSegment (speaker, text, timestamp)"]
    end

    subgraph Storage["5. Storage & Search (src/store.py)"]
        TaggedSeg --> SQLite["TranscriptStore (SQLite data/snapscribe.db)"]
        SQLite --> FTS["segments_fts (FTS5 Full-Text Index)"]
    end

    subgraph Summarization["6. Summarization (src/summarize.py)"]
        SQLite -.->|Periodic Query (every N min)| Rolling["RollingSummarizer (Phi-3.5-mini-instruct)"]
        Rolling -.->|Action Items & Summary| SQLite
    end

    subgraph UI["7. User Interface (src/ui_overlay.py)"]
        TaggedSeg --> Caption["CaptionOverlay (Tkinter Topmost Window)"]
        MainCtrl["Tray Icon / Lifecycle"] --> Tray["pystray System Tray Icon"]
    end
```

### Stage Details

1. **Audio Capture (`src/audio_capture.py`)**:
   - Captures two separate logical channels via `pyaudiowpatch`:
     - `"mic"`: captures local user speech from the default input device.
     - `"loopback"`: captures remote call participants via Windows WASAPI loopback from the default audio output endpoint.
   - Pushes raw 16 kHz 16-bit mono PCM chunks into a shared, thread-safe `queue.Queue`.
2. **Voice Activity Detection (`src/vad.py`)**:
   - `SpeechSegmenter` processes incoming 30ms audio frames (480 samples at 16 kHz) using WebRTC VAD (`webrtcvad.Vad(aggressiveness=2)`).
   - Buffers speech per channel and drops silence. When silence duration exceeds `silence_timeout_ms` (default 800ms / ~26 frames), the buffered segment is flushed as a complete speech segment.
3. **ASR / Transcription (`src/transcribe.py`)**:
   - Converts raw int16 PCM bytes to normalized float32 arrays `[-1.0, 1.0]`.
   - Invokes `qai_hub_models` `whisper_base` (`HfWhisperApp` wrapping `WhisperBase` encoder/decoder models) or fallback ONNX Runtime inference sessions.
4. **Speaker Tagging / Diarization (`src/diarize.py`)**:
   - In v1, channel separation provides zero-cost speaker attribution:
     - `"mic"` $\rightarrow$ `"You"`
     - `"loopback"` $\rightarrow$ `"Them"`
   - Emits a `TaggedSegment` with attributed speaker, transcribed text, and epoch timestamp.
5. **Storage & Full-Text Search (`src/store.py`)**:
   - Stores segments and rolling summaries in a local SQLite database (`data/snapscribe.db`).
   - Uses an SQLite `FTS5` virtual table (`segments_fts`) with an automated `AFTER INSERT` SQL trigger to enable fast full-text keyword and phrase search without cloud dependencies.
6. **Rolling Summarization (`src/summarize.py`)**:
   - Invokes Microsoft's `Phi-3.5-mini-instruct` (INT4 quantized) via `onnxruntime-genai`.
   - Periodically queries recent session transcripts and prompts the LLM to extract a 2-3 sentence overview and bulleted action items.
7. **Overlay UI (`src/ui_overlay.py`)**:
   - Displays real-time rolling captions in a borderless, semi-transparent, always-on-top Tkinter window.
   - Maintains a system tray icon with a recording indicator and quit controls via `pystray`.

---

## 2. Threading Model

SnapScribe runs as an asynchronous, multi-threaded application orchestrated by `src/main.py`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Main Thread (src/main.py)                        │
│  - Loads config and initializes store, transcriber, summarizer, segmenter  │
│  - Spawns background worker threads                                         │
│  - Manages shutdown signal loop (time.sleep / KeyboardInterrupt)            │
└──────┬────────────────────┬────────────────────┬─────────────────────┬──────┘
       │                    │                    │                     │
       ▼                    ▼                    ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Capture      │     │ Transcription│     │ Summarization│     │ UI Threads   │
│ Thread(s)    │     │ Thread       │     │ Thread       │     │ (Overlay +   │
│              │     │              │     │              │     │  Tray)       │
│ WASAPI Stream│     │ Reads queue, │     │ Periodic     │     │              │
│ callbacks    │     │ runs VAD,    │     │ interval     │     │ Tkinter      │
│ feed frames  │     │ ASR model,   │     │ timer, runs  │     │ mainloop &   │
│ into         │     │ tags speaker,│     │ Phi-3.5,     │     │ pystray tray │
│ Queue        │     │ writes store │     │ updates store│     │ event loop   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Main Orchestration Thread (`SnapScribeApp.start`)**:
   - Loads model weights and initializes SQLite database connections.
   - Launches capture streams and spawns background worker threads.
   - Monitors `_stop_event` until interrupted (e.g. via Ctrl+C or Tray Quit).
2. **Audio Capture Callbacks (PyAudioWPatch worker threads)**:
   - PortAudio/WASAPI invokes asynchronous stream callbacks on audio hardware interrupts.
   - Non-blocking callbacks place `(channel_label, pcm_bytes)` tuples onto `AudioCapture._frames` (`queue.Queue`).
3. **Transcription Worker Thread (`_transcription_loop`)**:
   - Consumes frames from the audio queue.
   - Passes frames through `SpeechSegmenter`.
   - When a segment boundary is detected, executes Whisper transcription, tags the speaker, commits to `TranscriptStore`, and dispatches the caption to the overlay.
4. **Summarization Worker Thread (`_summarization_loop`)**:
   - Runs independently on a configurable timer interval (`interval_minutes`).
   - Retrieves cumulative transcript text from `TranscriptStore`, executes `RollingSummarizer.summarize()`, and saves generated summaries back to the database.
5. **UI Threads (`CaptionOverlay` and `make_tray_icon`)**:
   - `CaptionOverlay.run()` executes the Tkinter `mainloop()` on its own daemon thread. Caption updates from the transcription loop are scheduled via `root.after(0, ...)` to ensure thread safety.
   - `tray.run()` runs the `pystray` system tray loop on a separate daemon thread to handle tray interactions and application termination.

---

## 3. Compute Allocation: Why NPU vs. CPU for Each Stage

| Stage | Module | Target Device | Execution Provider | Engineering Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Audio Capture** | `src/audio_capture.py` | **CPU** | Native OS Audio I/O | Standard Windows WASAPI buffer streaming. Negligible CPU utilization (<0.5%). |
| **Voice Activity Detection** | `src/vad.py` | **CPU** | WebRTC VAD (C extension) | WebRTC VAD uses lightweight integer Gaussian Mixture Models (<0.1% CPU). Gating the pipeline on CPU prevents the NPU from spinning up on silence, saving significant battery. |
| **Speech-to-Text (ASR)** | `src/transcribe.py` | **Hexagon NPU** *(CPU fallback in dev)* | `QNNExecutionProvider` *(Hexagon NPU)* | Continuous audio transcription involves sustained matrix multiplications and attention layers. The Hexagon NPU (45 TOPS on Snapdragon X Elite) offers superior TOPS/watt efficiency, enabling all-day transcription without CPU thermal throttling or battery drain. |
| **Diarization** | `src/diarize.py` | **CPU** | Heuristic channel routing | Hardware-isolated capture ("mic" vs. "loopback") allows instantaneous $O(1)$ string tagging without needing heavy clustering models (e.g., ECAPA-TDNN), conserving NPU budget for ASR and LLM inference. |
| **Storage & Search** | `src/store.py` | **CPU** | SQLite + FTS5 C-Engine | B-Tree storage and full-text inverted index searches are I/O and string operations that run in single-digit milliseconds on CPU; no neural tensor operations required. |
| **Summarization** | `src/summarize.py` | **Hexagon NPU** *(CPU fallback in dev)* | `QNNExecutionProvider` via `onnxruntime-genai` | Autoregressive token generation for a 3.8B parameter LLM (Phi-3.5) requires high compute and memory bandwidth. Running on NPU INT4 delivers fast generation while keeping CPU free for system responsiveness. |
| **UI & Tray** | `src/ui_overlay.py` | **CPU** | Tkinter / Win32 GUI | Rendering borderless text overlays and managing Windows shell notification icons are standard desktop GUI events requiring no ML acceleration. |
