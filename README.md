# proai-hf-export-starter

Export a small Hugging Face chat model to **ONNX** and **OpenVINO** with the exporters built into
Transformers 5.23. Then reload the saved export from disk, check it against the original PyTorch model,
and generate text with it.

This is the code from the ProAI Clips video on exporting Hugging Face models (video link coming soon).

- Model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0, public, no token needed),
  pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca` (`--revision`; pass `--revision main` if you use a different `--model`).
- Runs on a plain CPU. No GPU and no API keys.
- Docs this follows: [Transformers exporters](https://huggingface.co/docs/transformers/main/en/exporters).

> The exporters are experimental. In the docs' words: "Pin those versions in production tooling".
> `requirements.txt` has the exact versions we tested with.

## Before vs now

**Before:** you exported with a separate library, [Optimum](https://huggingface.co/docs/optimum-onnx/onnx/overview).
It only knows the architectures on its supported list. A model that isn't on the list fails with
`... is a custom or unsupported architecture`. Examples:
[Gemma 3](https://github.com/huggingface/optimum/issues/2371),
[SmolVLM2](https://github.com/huggingface/optimum/issues/2431) and
[Hunyuan](https://github.com/huggingface/optimum/issues/2433).
To fix that, you waited for support or wrote an export config yourself. Optimum still exists and is still documented.

**Now:** the exporters live inside Transformers. They first shipped in
[v5.13.0](https://github.com/huggingface/transformers/releases/tag/v5.13.0) (3 Jul 2026). From the docs:

> "The exporters live inside Transformers instead of a downstream library, so architecture changes, new
> attention patterns, and custom cache types are supported at export time as soon as they land in the
> modeling code." ([source](https://huggingface.co/docs/transformers/main/en/exporters))

In [v5.23.0](https://github.com/huggingface/transformers/releases/tag/v5.23.0) (13 Nov 2026), every export
returns an `ExportedModel`. You can `save_pretrained` it, load it back with `ExportedModel.from_pretrained`,
and call it like the original model.

## Quick start

### 1. Install (pinned)
```bash
git clone https://github.com/schoolofai/proai-hf-export-starter
cd proai-hf-export-starter
python3 -m venv venv
./venv/bin/pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt
```

### 2. Export to ONNX, reload, generate
How it's built ([`export_and_check.py`](export_and_check.py)):
```python
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", dtype=torch.float32)
exported = OnnxExporter().export(model, dict(inputs), config=OnnxConfig(dynamic=True))
exported.save_pretrained("exports/Qwen3-0.6B-onnx")
loaded = ExportedModel.from_pretrained("exports/Qwen3-0.6B-onnx")
logits = loaded(input_ids=..., attention_mask=...).logits
```
Run it:
```bash
./venv/bin/python export_onnx.py
```
Output from our run:
```
saved   exports/Qwen3-0.6B-onnx  (2.39 GB)
max logit diff vs original: 8.32e-05   same tokens: True
exported model says: 1. Use clear variable names and avoid unnecessary whitespace.
2. Add comments where necessary to explain complex logic.
```

### 3. Export to OpenVINO, reload, generate
The flow is the same with `OpenVINOExporter()` and `OpenVINOConfig(dynamic=True)`. The reload pins f32 (see the gotchas):
```python
loaded = ExportedModel.from_pretrained("exports/Qwen3-0.6B-openvino", device="cpu",
                                       config={"INFERENCE_PRECISION_HINT": "f32"})
```
```bash
./venv/bin/python export_openvino.py
```
Output from our run:
```
saved   exports/Qwen3-0.6B-openvino  (2.39 GB)
max logit diff vs original: 1.13e-04   same tokens: True
exported model says: 1. Use clear variable names and avoid unnecessary whitespace.
2. Add comments where necessary to explain complex logic.
```

### 4. Compare a saved export with the original (no re-export)
```bash
./venv/bin/python compare_outputs.py exports/Qwen3-0.6B-onnx
./venv/bin/python compare_outputs.py exports/Qwen3-0.6B-openvino --ov-precision default
```
The second command lets OpenVINO pick its own precision. On our CPU it picked bf16:
```
openvino precision: <Type: 'bfloat16'>
max logit diff vs original: 1.59e+01
same tokens as original: False
exported: 1. Use descriptive variable names to improve readability.
2. Add comments where necessary to explain complex logic.
```

The full results of our runs are in [`results/`](results/).

## What this is, and what it isn't
- The export is **one core graph**: tokens in, next-token scores (logits) out, with no KV cache.
  The text comes from **our own simple loop** in [`common.py`](common.py), which re-runs the whole sequence
  for every new token. That's fine for checking an export. It is not a fast chat app, and it isn't `generate()`.
- **Multi-part generation exports don't run yet.** The docs say: "Exports with several components
  (generation) don't run yet." `export_for_generation` gives you separate prefill and decode graphs that you
  have to orchestrate yourself.
- **If you just want to chat with a model on your laptop,** use [llama.cpp](https://github.com/ggml-org/llama.cpp)
  with a GGUF file.
- **Size:** each export folder is about 2.39 GB, because the weights are stored in float32.

## Gotchas
- **Use `sdpa` attention.** FlashAttention and FlexAttention are not exportable on any backend (docs).
- **OpenVINO precision (in our test).** The export records f32. Even so, on our Intel Xeon CPU with
  bf16 support, OpenVINO ran it in bf16 unless we passed `INFERENCE_PRECISION_HINT: f32`, and the
  generated text changed. The scripts pin f32 by default.
- **Upgrading from older 5.x exporter code?** In 5.23, `export()` returns an `ExportedModel`, and the `output_path`
  config option is gone in favour of `save_pretrained`
  ([release notes](https://github.com/huggingface/transformers/releases/tag/v5.23.0)).
- **Timings.** `results/*.json` holds one timed run on our CPU (8 vCPU Intel Xeon), for example
  ONNX export 43.8 s and OpenVINO export 24.7 s (the run shown in the video). They come from a single run on our CPU and are not benchmarks.

## Files
| File | What it does |
|---|---|
| `common.py` | Loads the model (sdpa, pinned revision), builds the chat prompt, runs the simple greedy loop |
| `export_and_check.py` | Exports, saves, reloads, then compares with the original and generates |
| `export_onnx.py` / `export_openvino.py` | Run that flow for one backend |
| `compare_outputs.py` | Reloads a saved export and compares it with the original model |
| `requirements.txt` | Exact versions we tested |

Options: `--model`, `--prompt`, `--max-new-tokens`, and `--ov-precision` (`f32`, `bf16`, `f16` or `default`).
Other small chat models may work too. We tested Qwen3-0.6B and SmolLM2-135M-Instruct.

## License
MIT for the code in this repo. The model weights keep their own licence (Qwen3-0.6B: Apache-2.0).
