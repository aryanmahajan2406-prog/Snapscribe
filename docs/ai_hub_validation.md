# NPU Validation via Qualcomm AI Hub

Until physical Snapdragon HP hardware was available, this project validated
real NPU performance remotely against a **cloud-hosted "Snapdragon X Elite
CRD"** device through Qualcomm AI Hub (`scripts/export_models.py`).

## How to fill this in

1. Run `python scripts/export_models.py`
2. Open the printed profile job link on aihub.qualcomm.com
3. Copy the reported numbers into the table below

| Model              | Device               | Latency (ms) | Peak memory | Compute unit |
|---------------------|------------------------|---------------|--------------|----------------|
| Whisper Base (EN)   | Snapdragon X Elite CRD | _fill in_     | _fill in_    | NPU            |
| Phi-3.5-mini-instruct (QNN, INT4) | Snapdragon X2 Elite | _fill in_ | _fill in_ | NPU |

## Once physical hardware is available

Re-run the same pipeline with `execution_provider: QNNExecutionProvider` in
`config/settings.yaml` and record on-device battery/thermal numbers here too
— these are the numbers that matter most for the "feels integrated,
all-day-usable" pitch.
