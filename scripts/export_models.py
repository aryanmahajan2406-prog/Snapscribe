"""
Compiles, profiles, and downloads SnapScribe's Whisper model via Qualcomm AI
Hub, targeting a real cloud-hosted Snapdragon X Elite reference device.

This shells out to qai_hub_models' own per-model export CLI
(`python -m qai_hub_models.models.<model_id>.export`), which is the actual
supported entry point for compiling + profiling in this package version --
confirmed working after testing against the real AI Hub Workbench.

Prereqs:
    pip install "qai_hub_models[whisper_base]"
    qai-hub configure --api_token <YOUR_TOKEN>

Note on model name: earlier drafts of this script referenced
"whisper_base_en", which doesn't exist in this qai_hub_models release.
The real folder name is "whisper_base" (confirmed via check_whisper.py).
"""

import subprocess
import sys

DEVICE = "Snapdragon X Elite CRD"  # cloud-hosted reference device on AI Hub


def list_snapdragon_devices():
    """Run this first if DEVICE below throws an 'unrecognized device' error."""
    import qai_hub as hub

    for d in hub.get_devices():
        if "Snapdragon" in d.name or "Elite" in d.name:
            print(d.name)


def export_whisper():
    cmd = [
        sys.executable,
        "-m",
        "qai_hub_models.models.whisper_base.export",
        "--device",
        DEVICE,
        # AI Hub Workbench has moved past the QAIRT version this
        # qai_hub_models release defaults to -- pin to "default" explicitly.
        "--compile-options",
        "--qairt_version=default",
        "--profile-options",
        "--qairt_version=default",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_summarizer_note():
    # Phi-3.5-mini-instruct QNN builds are published directly by Microsoft on
    # Hugging Face (microsoft/Phi-3.5-mini-instruct-onnx, "qnn" subfolder),
    # already compiled for Snapdragon -- no AI Hub compile step needed, just
    # download into models/phi-3.5-mini-instruct-onnx/qnn/.
    print(
        "\nSummarizer: download microsoft/Phi-3.5-mini-instruct-onnx (qnn "
        "subfolder) from Hugging Face into "
        "models/phi-3.5-mini-instruct-onnx/qnn/"
    )


def main():
    export_whisper()
    export_summarizer_note()
    print(
        "\nDone. Open the printed AI Hub job link(s) on aihub.qualcomm.com "
        "and copy the latency / memory / compute-unit numbers into "
        "docs/ai_hub_validation.md for the pitch deck."
    )


if __name__ == "__main__":
    main()
