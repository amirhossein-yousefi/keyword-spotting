from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import torch

# Reuse the robust dataset loader + preprocess that work without TorchCodec/Audio()
from src.data_utils import load_speech_commands, make_preprocess_fn

def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoint on Speech Commands test set")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_name", default="speech_commands")
    ap.add_argument("--dataset_config", default="v0.02")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--max_duration_seconds", type=float, default=1.0)
    ap.add_argument("--report_path", type=str, default=None)
    ap.add_argument("--cm_path", type=str, default=None)
    args = ap.parse_args()

    # Load dataset using the same resilient path as training
    ds = load_speech_commands(args.dataset_name, args.dataset_config, sample_rate=args.sample_rate)

    fe = AutoFeatureExtractor.from_pretrained(args.checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(args.checkpoint)

    # Preprocess function (pads/trims to fixed 1s window, handles augment flag off)
    prep = make_preprocess_fn(
        fe,
        sample_rate=args.sample_rate,
        max_duration_seconds=args.max_duration_seconds,
        is_training=False,
        augment_cfg={"enabled": False},
    )

    test = ds["test"].map(prep, batched=True, remove_columns=ds["test"].column_names).with_format("torch")

    # Lightweight Trainer just for prediction
    dummy_args = TrainingArguments(
        output_dir=os.path.join(args.checkpoint, "_eval_tmp"),
        report_to=["none"],
        per_device_eval_batch_size=32,
    )
    trainer = Trainer(model=model, args=dummy_args, data_collator=default_data_collator)

    preds = trainer.predict(test)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)

    # Get label names robustly
    id2label = getattr(model.config, "id2label", None) or {}
    try:
        # Keys may be strings in config JSON; normalize to int
        id2label = {int(k): v for k, v in id2label.items()}
    except Exception:
        pass
    num_labels = getattr(model.config, "num_labels", len(set(y_true)))
    target_names = [id2label.get(i, str(i)) for i in range(num_labels)]

    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report)

    if args.report_path:
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            f.write(report)

    if args.cm_path:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        os.makedirs(os.path.dirname(args.cm_path), exist_ok=True)
        fig.savefig(args.cm_path, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
