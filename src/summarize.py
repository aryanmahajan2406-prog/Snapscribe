"""
Rolling summarization via Phi-3.5-mini-instruct, running on the Hexagon NPU
through onnxruntime-genai (the Microsoft toolkit purpose-built for on-device
LLM inference on Snapdragon Copilot+ PCs).

Model: microsoft/Phi-3.5-mini-instruct-onnx, "qnn" subfolder (QNN-optimized,
INT4-quantized weights) — download instructions in scripts/export_models.py.
"""

from pathlib import Path

SUMMARY_PROMPT = """You are summarizing a live meeting transcript. Given the
transcript so far, produce:
1. A 2-3 sentence rolling summary of what's been discussed.
2. Any action items mentioned, as a short bullet list (or "None" if none).

Transcript:
{transcript}

Respond in this format:
Summary: ...
Action items:
- ...
"""


class RollingSummarizer:
    def __init__(self, model_dir: str, max_tokens: int = 200):
        self.model_dir = Path(model_dir)
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def load(self):
        import onnxruntime_genai as og

        self._model = og.Model(str(self.model_dir))
        self._tokenizer = og.Tokenizer(self._model)

    def summarize(self, transcript_text: str) -> str:
        if self._model is None:
            raise RuntimeError("Call .load() before .summarize()")

        import onnxruntime_genai as og

        prompt = SUMMARY_PROMPT.format(transcript=transcript_text)
        input_tokens = self._tokenizer.encode(prompt)

        params = og.GeneratorParams(self._model)
        params.set_search_options(max_length=self.max_tokens, temperature=0.3)
        params.input_ids = input_tokens

        generator = og.Generator(self._model, params)
        output_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        return self._tokenizer.decode(output_tokens)
