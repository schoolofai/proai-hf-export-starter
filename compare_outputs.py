"""Reload a saved export (no re-export) and compare it with the original PyTorch model.

    python compare_outputs.py exports/Qwen3-0.6B-onnx
    python compare_outputs.py exports/Qwen3-0.6B-openvino
    python compare_outputs.py exports/Qwen3-0.6B-openvino --ov-precision default
"""
import json
from pathlib import Path

from transformers.exporters import ExportedModel

from common import chat_inputs, greedy, load, max_logit_diff, openvino_load_options, parse_args

args = parse_args(lambda ap: ap.add_argument("export_dir", type=Path))
backend = json.loads((args.export_dir / "export.json").read_text())["export_format"]
model, tok, _ = load(args.model)
inputs = chat_inputs(tok, args.prompt)

options = openvino_load_options(args.ov_precision) if backend == "openvino" else {}
loaded = ExportedModel.from_pretrained(args.export_dir, **options)

diff = max_logit_diff(model, loaded, inputs)
original_text, original_ids = greedy(model, inputs, tok, args.max_new_tokens)
exported_text, exported_ids = greedy(loaded, inputs, tok, args.max_new_tokens)

print(f"backend: {backend}")
if backend == "openvino":
    runner = next(iter(loaded.runners.values()))
    print(f"openvino precision: {runner.compiled_model.get_property('INFERENCE_PRECISION_HINT')}")
print(f"max logit diff vs original: {diff:.2e}")
print(f"same tokens as original: {exported_ids == original_ids}")
print(f"original: {original_text}")
print(f"exported: {exported_text}")
