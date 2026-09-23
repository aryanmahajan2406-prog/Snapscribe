"""
Compiles and profiles SnapScribe's models on a real, cloud-hosted Snapdragon
device via Qualcomm AI Hub — no physical hardware required for this step.

Prereqs:
    pip install qai-hub qai_hub_models
    Sign up at https://aihub.qualcomm.com, then:
        qai-hub configure --api_token <YOUR_TOKEN>

This produces:
    - Compiled QNN-targeted ONNX models (downloaded locally, ready to bundle)
    - Real latency / memory / NPU-utilization numbers from the cloud-hosted
      device (use these in the pitch deck / docs/ai_hub_validation.md)
"""

import qai_hub as hub

DEVICE = "Snapdragon X Elite CRD"  # cloud-hosted reference device on AI Hub


def export_whisper():
    from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn

    print(f"Compiling Whisper Base (EN) for {DEVICE}...")
    model = WhisperBaseEn.from_pretrained()

    compile_job = hub.submit_compile_job(
        model=model,
        device=hub.Device(DEVICE),
        options="--target_runtime onnx",
    )
    target_model = compile_job.get_target_model()

    print("Profiling on cloud-hosted device...")
    profile_job = hub.submit_profile_job(model=target_model, device=hub.Device(DEVICE))
    profile_job.wait()

    print("Downloading compiled model to models/whisper_base_en/ ...")
    target_model.download("models/whisper_base_en")

    return profile_job


def export_summarizer_note():
    # Phi-3.5-mini-instruct QNN builds are published directly by Microsoft on
    # Hugging Face (microsoft/Phi-3.5-mini-instruct-onnx, "qnn" subfolder),
    # already compiled for Snapdragon — no AI Hub compile step needed, just
    # download into models/phi-3.5-mini-instruct-onnx/qnn/ and point
    # config/settings.yaml at it.
    print(
        "Summarizer: download microsoft/Phi-3.5-mini-instruct-onnx (qnn "
        "subfolder) from Hugging Face into "
        "models/phi-3.5-mini-instruct-onnx/qnn/"
    )


def main():
    profile_job = export_whisper()
    export_summarizer_note()

    print("\nDone. Record the profile_job numbers (latency, memory, NPU vs "
          "CPU/GPU split) into docs/ai_hub_validation.md for the pitch deck.")
    print(f"Profile job: {profile_job}")


if __name__ == "__main__":
    main()
