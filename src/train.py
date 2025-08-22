from __future__ import annotations
import argparse, os, json, yaml
from typing import Dict, Any
import numpy as np
import evaluate
import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from src.utils import set_seed, ensure_dir
from src.data_utils import (
    load_speech_commands,
    build_feature_extractor,
    make_preprocess_fn,
    label_maps,
    build_data_collator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Keyword Spotting with HF Transformers")
    p.add_argument("--config", default="../configs/train_config.yaml", type=str, help="YAML config file")
    p.add_argument("--output_dir", default="./runs/wav2vec2-kws", type=str, help="Where to store checkpoints & logs")
    # Optional CLI overrides
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--model_name_or_path", type=str, default=None)
    return p.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    ensure_dir(args.output_dir)

    # Allow CLI overrides for common params
    for k in ["learning_rate", "num_train_epochs", "train_batch_size", "eval_batch_size", "model_name_or_path"]:
        v = getattr(args, k, None)
        if v is not None:
            # map to config keys
            if k == "train_batch_size":
                cfg["train_batch_size"] = v
            elif k == "eval_batch_size":
                cfg["eval_batch_size"] = v
            else:
                cfg[k] = v

    set_seed(cfg["seed"])

    # Data
    ds: DatasetDict = load_speech_commands(
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg["dataset_config"],
        sample_rate=cfg["sample_rate"],
        subset_fraction=cfg.get("subset_fraction", 1.0),
    )
    labels, id2label, label2id = label_maps(ds)

    # Feature extractor & preprocess
    fe = build_feature_extractor(cfg["model_name_or_path"], cfg["sample_rate"])
    prep_train = make_preprocess_fn(
        fe,
        sample_rate=cfg["sample_rate"],
        max_duration_seconds=cfg["max_duration_seconds"],
        is_training=True,
        augment_cfg=cfg.get("augment", {"enabled": False}),
    )
    prep_eval = make_preprocess_fn(
        fe,
        sample_rate=cfg["sample_rate"],
        max_duration_seconds=cfg["max_duration_seconds"],
        is_training=False,
        augment_cfg={"enabled": False},
    )
    ds_proc = DatasetDict({
        "train": ds["train"].map(prep_train, batched=True, remove_columns=ds["train"].column_names),
        "validation": ds["validation"].map(prep_eval, batched=True, remove_columns=ds["validation"].column_names),
        "test": ds["test"].map(prep_eval, batched=True, remove_columns=ds["test"].column_names),
    })
    # Set format to torch
    ds_proc = DatasetDict({k: v.with_format("torch") for k, v in ds_proc.items()})

    # Model
    model = AutoModelForAudioClassification.from_pretrained(
        cfg["model_name_or_path"],
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
    )

    # Metrics
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=1)
        return {
            "accuracy": acc.compute(predictions=y_pred, references=y_true)["accuracy"],
            "f1_weighted": f1.compute(predictions=y_pred, references=y_true, average="weighted")["f1"],
            "precision_weighted": prec.compute(predictions=y_pred, references=y_true, average="weighted")["precision"],
            "recall_weighted": rec.compute(predictions=y_pred, references=y_true, average="weighted")["recall"],
        }

    is_windows = (os.name == "nt")
    num_workers = 0 if is_windows else 4  # <- disable workers on Windows
    # Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["num_train_epochs"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=cfg.get("fp16", False),
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["evaluation_strategy"],
        save_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        push_to_hub=cfg.get("push_to_hub", False),
        report_to=["tensorboard"],
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=False,
        logging_dir="logs/training-logs",

    )

    data_collator = build_data_collator(fe)

    callbacks = []
    # Early stopping after a few evals without improvement
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        tokenizer=fe,  # for saving processor
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-best"))
    fe.save_pretrained(os.path.join(args.output_dir, "checkpoint-best"))

    # Save train metrics
    with open(os.path.join(args.output_dir, "train_results.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Final evaluation on validation
    val_metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    # Also evaluate on test
    test_metrics = trainer.evaluate(ds_proc["test"])
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Training complete.")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
