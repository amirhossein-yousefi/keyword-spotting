# src/stream_infer.py
from __future__ import annotations
import argparse, queue, sys, time
import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def main():
    ap = argparse.ArgumentParser(description="Realtime streaming KWS from microphone")
    ap.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    ap.add_argument("--window_ms", type=int, default=1000, help="analysis window (ms)")
    ap.add_argument("--hop_ms", type=int, default=200, help="hop size (ms) between windows")
    ap.add_argument("--smooth", type=int, default=5, help="moving-average smoothing over N windows")
    ap.add_argument("--prob_threshold", type=float, default=0.8, help="trigger probability")
    ap.add_argument("--target_keywords", nargs="*", default=None, help="list of keywords to watch; if None, all")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained(args.checkpoint).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(args.checkpoint)
    sr = getattr(fe, "sampling_rate", 16000)
    id2label = model.config.id2label
    label2id = {v: int(k) for k, v in model.config.id2label.items()}

    window_samples = int(args.window_ms * sr / 1000)
    hop_samples = int(args.hop_ms * sr / 1000)
    buffer = np.zeros(window_samples, dtype=np.float32)

    q = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=hop_samples,
        callback=audio_cb,
    )

    smooth_probs = None
    smooth_len = args.smooth

    watched = None
    if args.target_keywords:
        watched = set(args.target_keywords)

    print("Starting mic stream. Press Ctrl+C to stop.")
    with stream:
        while True:
            block = q.get()
            block = block.squeeze(-1) if block.ndim > 1 else block
            # Slide buffer and append new block
            if len(block) < hop_samples:
                block = np.pad(block, (0, hop_samples - len(block)))
            buffer = np.roll(buffer, -hop_samples)
            buffer[-hop_samples:] = block[:hop_samples]

            # Run model
            inputs = fe([buffer], sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits[0].cpu().numpy()
            probs = softmax(logits)

            # Smooth
            if smooth_probs is None:
                smooth_probs = np.zeros((smooth_len, probs.shape[0]), dtype=np.float32)
            smooth_probs = np.roll(smooth_probs, -1, axis=0)
            smooth_probs[-1] = probs
            mean_probs = smooth_probs.mean(axis=0)

            top_idx = int(mean_probs.argmax())
            top_label = id2label[top_idx]
            top_prob = float(mean_probs[top_idx])

            if (watched is None or top_label in watched) and top_prob >= args.prob_threshold:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] DETECTED: {top_label} (p={top_prob:.2f})")

