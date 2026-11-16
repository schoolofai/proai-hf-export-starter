"""Export a Hugging Face chat model to OpenVINO IR, reload it on the CPU, and generate.

    python export_openvino.py --model Qwen/Qwen3-0.6B
"""
from export_and_check import main

if __name__ == "__main__":
    main("openvino")
