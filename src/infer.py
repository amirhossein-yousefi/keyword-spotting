from __future__ import annotations
import argparse, os, json
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

def load_audio(path: str, target_sr: int) -> np.ndarray:
    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = np.mean(wave, axis=1)  # mono
    if sr != target_sr:
        # lightweight resample with numpy-based polyphase (fallback if librosa not installed)
        import math
        import numpy as np
        # Simple linear resampler (OK for inference demo)
        duration = len(wave) / sr
        new_len = int(round(duration * target_sr))
        new_idx = np.linspace(0, len(wave)-1, new_len)
        wave = np.interp(new_idx, np.arange(len(wave)), wave).astype(np.float32)
    return wave

def main():
    ap = argparse.ArgumentParser(description="Offline file inference for KWS")
    ap.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (dir with config.json)")
    ap.add_argument("--files", nargs="+", required=True, help="List of audio files (.wav recommended)")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained(args.checkpoint).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(args.checkpoint)
    sr = getattr(fe, "sampling_rate", 16000)
    id2label = model.config.id2label

    for path in args.files:
        wave = load_audio(path, target_sr=sr)
        # One-second window (pad/trim)
        max_len = sr  # 1s
        if len(wave) < max_len:
            wave = np.pad(wave, (0, max_len - len(wave)))
        else:
            wave = wave[:max_len]

        inputs = fe([wave], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_idx = probs.argsort()[::-1][:args.top_k]
        print(f"\nFile: {os.path.basename(path)}")
        for i in top_idx:
            print(f"  {id2label[i]:>10s} : {probs[i]:.3f}")

if __name__ == "__main__":
    main()
