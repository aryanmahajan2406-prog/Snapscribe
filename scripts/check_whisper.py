"""
Diagnostic: find out what version of qai_hub_models is installed and which
Whisper model folders it actually ships, so we can match export_models.py
to the real module path.

Run with: python check_whisper.py
"""

import os

import qai_hub_models

print(f"qai_hub_models version : {qai_hub_models.__version__}")
print(f"qai_hub_models path    : {os.path.dirname(qai_hub_models.__file__)}")

models_dir = os.path.join(os.path.dirname(qai_hub_models.__file__), "models")

if not os.path.isdir(models_dir):
    print(f"No 'models' folder found at {models_dir} -- unexpected install layout.")
else:
    all_models = sorted(os.listdir(models_dir))
    whisper_models = [m for m in all_models if "whisper" in m.lower()]
    print(f"\nTotal model folders found: {len(all_models)}")
    print(f"Whisper-related folders  : {whisper_models}")

    if not whisper_models:
        print(
            "\nNo whisper folders at all -- this version may not ship Whisper "
            "as a model, or it's under a different name. Showing first 30 "
            "model folders so we can spot it manually:"
        )
        print(all_models[:30])
