import os
from importlib.metadata import version, PackageNotFoundError
import qai_hub_models

try:
    pkg_version = version("qai-hub-models")
except PackageNotFoundError:
    pkg_version = "unknown"

print("qai_hub_models version:", pkg_version)
print("qai_hub_models path:", os.path.dirname(qai_hub_models.__file__))

models_dir = os.path.join(os.path.dirname(qai_hub_models.__file__), "models")

if not os.path.isdir(models_dir):
    print("No models folder found at", models_dir)
else:
    all_models = sorted(os.listdir(models_dir))
    whisper_models = [m for m in all_models if "whisper" in m.lower()]
    print("Total model folders found:", len(all_models))
    print("Whisper-related folders:", whisper_models)