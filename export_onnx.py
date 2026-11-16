"""Export a Hugging Face chat model to ONNX, reload it with ONNX Runtime, and generate.

    python export_onnx.py --model Qwen/Qwen3-0.6B
"""
from export_and_check import main

if __name__ == "__main__":
    main("onnx")
