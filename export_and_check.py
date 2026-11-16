"""Export -> save -> reload -> compare with the original model. Used by export_onnx.py and export_openvino.py."""
import resource
import time
from pathlib import Path

from transformers.exporters import ExportedModel, OnnxConfig, OnnxExporter, OpenVINOConfig, OpenVINOExporter

from common import (chat_inputs, file_sizes, greedy, load, max_logit_diff, openvino_load_options,
                    parse_args, save_json)

EXPORTERS = {
    "onnx": lambda: (OnnxExporter(), OnnxConfig(dynamic=True)),
    "openvino": lambda: (OpenVINOExporter(), OpenVINOConfig(dynamic=True)),
}


def main(backend):
    args = parse_args()
    model, tok, revision = load(args.model)
    inputs = chat_inputs(tok, args.prompt)
    out = Path(args.out) / f"{args.model.split('/')[-1]}-{backend}"

    # 1. Export. dynamic=True lets the graph take any sequence length, so it can keep generating.
    exporter, config = EXPORTERS[backend]()
    t0 = time.perf_counter()
    exported = exporter.export(model, dict(inputs), config=config)
    export_s = time.perf_counter() - t0

    # 2. Save: model file(s) + export.json (config, shapes, package versions) + config.json.
    exported.save_pretrained(out)

    # 3. Reload from disk, the way your app would.
    options = openvino_load_options(args.ov_precision) if backend == "openvino" else {}
    loaded = ExportedModel.from_pretrained(out, **options)

    # 4. Check it against the original model, then generate with our own loop.
    diff = max_logit_diff(model, loaded, inputs)
    original_text, original_ids = greedy(model, inputs, tok, args.max_new_tokens)
    t0 = time.perf_counter()
    exported_text, exported_ids = greedy(loaded, inputs, tok, args.max_new_tokens)
    loop_s = time.perf_counter() - t0

    result = {
        "backend": backend, "model": args.model, "revision": revision, "prompt": args.prompt,
        "export_dir": out.as_posix(), "files_bytes": file_sizes(out),
        "max_abs_logit_diff_vs_original": diff,
        "same_tokens_as_original": exported_ids == original_ids,
        "original_text": original_text, "exported_text": exported_text,
        "single_run_our_cpu": {
            "export_seconds": round(export_s, 1),
            "greedy_loop_seconds_no_kv_cache": round(loop_s, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        },
    }
    if backend == "openvino":
        runner = next(iter(loaded.runners.values()))
        result["openvino_inference_precision"] = str(runner.compiled_model.get_property("INFERENCE_PRECISION_HINT"))
    save_json(f"results/{out.name}.json", result)
    print(f"saved   {out}  ({sum(result['files_bytes'].values()) / 1e9:.2f} GB)")
    print(f"max logit diff vs original: {diff:.2e}   same tokens: {result['same_tokens_as_original']}")
    print(f"exported model says: {exported_text}")
