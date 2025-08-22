from __future__ import annotations
import argparse
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def main():
    ap = argparse.ArgumentParser(description="Export audio classifier to ONNX")
    ap.add_argument("--checkpoint", required=True, help="Path to trained model directory")
    ap.add_argument("--onnx_path", required=True, help="Output .onnx file")
    ap.add_argument("--sample_rate", type=int, default=None)
    ap.add_argument("--window_seconds", type=float, default=1.0)
    args = ap.parse_args()

    model = AutoModelForAudioClassification.from_pretrained(args.checkpoint).eval()
    fe = AutoFeatureExtractor.from_pretrained(args.checkpoint)
    sr = args.sample_rate or getattr(fe, "sampling_rate", 16000)
    dummy_len = int(sr * args.window_seconds)

    dummy = torch.tensor([np.zeros(dummy_len, dtype=np.float32)])
    # The HF feature extractor expects raw wave; we export the model accepting input_values directly.
    input_names = ["input_values"]
    output_names = ["logits"]

    torch.onnx.export(
        model,
        args=(dummy,),
        f=args.onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_values": {1: "time"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {args.onnx_path}")

if __name__ == "__main__":
    main()
