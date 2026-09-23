# Privacy

SnapScribe's entire value proposition is that nothing leaves the device:

- Audio is captured, transcribed, and discarded — never uploaded.
- Transcripts and summaries live in a local SQLite file
  (`data/snapscribe.db`), never synced anywhere.
- Model inference (ASR + summarization) runs on the local Hexagon NPU.
  Model *compilation/profiling* during development used Qualcomm AI Hub's
  cloud-hosted device for validation only — no meeting content was ever
  sent there, only the model weights themselves.
- No account, no login, no telemetry.

This is the direct contrast with cloud tools (Otter, Teams Premium, Zoom AI
Companion): same functionality, zero data leaves your laptop.
