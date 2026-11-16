"""Shared helpers: load the model, build a chat prompt, run a simple greedy loop."""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "Give two tips for writing readable Python code, in two short sentences."


def parse_args(extra=None):
    ap = argparse.ArgumentParser()
    if extra:
        extra(ap)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default="exports")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--ov-precision", default="f32",
                    help="OpenVINO INFERENCE_PRECISION_HINT when loading (f32, bf16, f16), or 'default' to let OpenVINO choose")
    return ap.parse_args()


def load(model_id):
    """Load the original PyTorch model. Use sdpa attention: FlashAttention and FlexAttention don't export."""
    revision = model_info(model_id).sha  # pin the exact commit; public model, no token needed
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, attn_implementation="sdpa", dtype=torch.float32
    )
    model.config.use_cache = False  # one graph in, logits out (no KV cache)
    model.eval()
    return model, tok, revision


def chat_inputs(tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False,  # Qwen3: answer directly; other chat templates ignore this
    )
    return tok(text, return_tensors="pt")


def openvino_load_options(precision):
    options = {"device": "cpu"}
    if precision != "default":
        options["config"] = {"INFERENCE_PRECISION_HINT": precision}
    return options


@torch.no_grad()
def greedy(model, inputs, tok, max_new_tokens):
    """Our own simple loop: run the whole sequence, take the most likely next token, repeat.
    No KV cache, so it is slow and only for checking. Works for the original model and the export."""
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    start = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = torch.as_tensor(model(input_ids=ids, attention_mask=mask).logits)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    new = ids[0, start:]
    return tok.decode(new, skip_special_tokens=True).strip(), new.tolist()


def max_logit_diff(model, exported, inputs):
    with torch.no_grad():
        ref = model(**inputs).logits
        got = torch.as_tensor(exported(**inputs).logits)
    return (ref - got).abs().max().item()


def file_sizes(path):
    return {f.name: f.stat().st_size for f in sorted(Path(path).iterdir()) if f.is_file()}


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
