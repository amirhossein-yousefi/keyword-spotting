# hf-kws — Keyword Spotting with Hugging Face + PyTorch
> A compact, end‑to‑end pipeline for training, evaluating, and deploying a Wav2Vec2‑based keyword‑spotting (KWS) model. Includes realtime/streaming inference and one‑click AWS SageMaker deployment.
## Highlights
- **Train** a Hugging Face audio classifier (e.g., `facebook/wav2vec2-base`) for keyword spotting on **Speech Commands v2**.
- **Evaluate** with saved JSON metrics (train/val/test) and visualize **loss/F1** curves from `assets/`.
- **Infer** on single files or **stream from microphone**.
- **(Optional)** **Export to ONNX**.
- **Deploy** to **AWS SageMaker** (realtime, *serverless*, batch transform) + a small client to invoke the endpoint.
- **CI**: Ruff lint + PyTest, optional pipeline trigger.
## 🚀 Model on Hugging Face

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Keyword--Spotting-yellow.svg)](https://huggingface.co/Amirhossein75/Keyword-Spotting)

<p align="center">
  <a href="https://huggingface.co/Amirhossein75/Keyword-Spotting">
    <img src="https://img.shields.io/badge/🤗%20View%20on%20Hugging%20Face-blueviolet?style=for-the-badge" alt="Hugging Face Repo">
  </a>
</p>

## Repository Layout
```
hf-kws/
├── README.md
├── requirements.txt
├── src/
│   ├── data_utils.py
│   ├── augment.py
│   ├── utils.py
│   ├── train.py
│   ├── infer.py
│   ├── stream_infer.py
│   ├── evaluate.py
│   └── export_onnx.py
├── configs/
│   └── train_config.yaml
├── sagemaker/
│   ├── launch_training.py
│   ├── deploy_realtime.py
│   ├── deploy_serverless.py
│   ├── batch_transform.py
│   ├── pipeline.py
│   └── code/
│       ├── inference.py
│       └── requirements.txt
├── client/
│   ├── invoke_realtime.py
│   └── sample.jsonl
├── .github/workflows/ci.yml
├── Makefile
├── requirements.txt              
└── README_SageMaker.md
```
This project fine-tunes a Wav2Vec2 audio classifier for **keyword spotting** on the
open-source **Speech Commands v2** dataset, then runs both offline and realtime streaming inference.



## Features
- ✅ Fine-tune `Wav2Vec2` (or any HF audio classifier) with 🤗 `Trainer`
- ✅ Robust audio augmentations (time-shift, noise, random gain)
- ✅ Realtime streaming inference from microphone with sliding-window smoothing
- ✅ Offline file-based inference (single file or batch)
- ✅ Evaluation + confusion matrix
- ✅ (Optional) Export to ONNX for deployment

## Setup
```bash
cd hf-kws
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train 
```bash
python -m src.train \
  --checkpoint facebook/wav2vec2-base \
  --output_dir ./checkpoints/kws_w2v2 \
  --num_train_epochs 8 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16

```
**Download** the finetuned weights with bellow hyperparameter from [here](https://drive.google.com/file/d/1Fi0re7HHcYu83alKxYUgk2zMMUSgaLiN/view?usp=sharing)

## 🚀 Training Command

To reproduce the default training run:

```bash
python train.py \
```
By default, the script uses the speech_commands dataset.

## ⚙️ Training & Data Hyperparameters (KWS with HF Transformers)

| **Category**                   | **Parameter**                 | **Source / Key**                  | **Default / Example**          | **Description**                                               |
| ------------------------------ | ----------------------------- | --------------------------------- | ------------------------------ | ------------------------------------------------------------- |
| **Config I/O**                 | `config`                      | CLI                               | `../configs/train_config.yaml` | YAML file with all training config                            |
|                                | `output_dir`                  | CLI                               | `./runs/wav2vec2-kws`          | Checkpoints, logs, and metrics path                           |
| **CLI overrides → cfg**        | `learning_rate`               | CLI→cfg                           | `None`                         | If provided, overrides `cfg.learning_rate`                    |
|                                | `num_train_epochs`            | CLI→cfg                           | `None`                         | Overrides `cfg.num_train_epochs`                              |
|                                | `train_batch_size`            | CLI→cfg                           | `None`                         | Overrides `cfg.train_batch_size`                              |
|                                | `eval_batch_size`             | CLI→cfg                           | `None`                         | Overrides `cfg.eval_batch_size`                               |
|                                | `model_name_or_path`          | CLI→cfg                           | `None`                         | Overrides `cfg.model_name_or_path`                            |
| **Randomness**                 | `seed`                        | `cfg.seed`                        | *(from YAML)*                  | Set with `set_seed(cfg["seed"])`                              |
| **Dataset**                    | `dataset_name`                | `cfg.dataset_name`                | *(from YAML)*                  | e.g., `speech_commands`                                       |
|                                | `dataset_config`              | `cfg.dataset_config`              | *(from YAML)*                  | Subset/version of dataset                                     |
|                                | `sample_rate`                 | `cfg.sample_rate`                 | *(from YAML)*                  | Target sampling rate                                          |
|                                | `subset_fraction`             | `cfg.subset_fraction`             | `1.0`                          | Downsample dataset fraction                                   |
| **Preprocessing**              | Feature extractor             | `cfg.model_name_or_path`          | *(from YAML)*                  | Built via `build_feature_extractor`                           |
|                                | `max_duration_seconds`        | `cfg.max_duration_seconds`        | *(from YAML)*                  | Trim/pad window per clip                                      |
|                                | `augment.enabled`             | `cfg.augment.enabled`             | `False`                        | Train-time augmentation toggle                                |
| **Labels**                     | `label2id` / `id2label`       | derived                           | —                              | Built from dataset via `label_maps`                           |
| **Model**                      | `model_name_or_path`          | `cfg.model_name_or_path`          | *(from YAML)*                  | HF audio classifier backbone (e.g., `facebook/wav2vec2-base`) |
|                                | `num_labels`                  | derived                           | —                              | `len(labels)` from dataset                                    |
|                                | `problem_type`                | fixed                             | `single_label_classification`  |                                                               |
| **Optimization**               | `learning_rate`               | `cfg.learning_rate`               | *(from YAML/override)*         | AdamW LR (HF Trainer default optimizer)                       |
|                                | `weight_decay`                | `cfg.weight_decay`                | *(from YAML)*                  | L2 regularization                                             |
|                                | `num_train_epochs`            | `cfg.num_train_epochs`            | *(from YAML/override)*         | Epoch budget                                                  |
|                                | `gradient_accumulation_steps` | `cfg.gradient_accumulation_steps` | *(from YAML)*                  | Accumulate gradients N steps                                  |
|                                | `lr_scheduler_type`           | `cfg.lr_scheduler_type`           | *(from YAML)*                  | e.g., `linear`, `cosine`                                      |
|                                | `warmup_ratio`                | `cfg.warmup_ratio`                | *(from YAML)*                  | Warmup fraction of total steps                                |
|                                | `max_grad_norm`               | `cfg.max_grad_norm`               | `1.0`                          | Gradient clipping                                             |
| **Batching**                   | `per_device_train_batch_size` | `cfg.train_batch_size`            | *(from YAML/override)*         | Per-GPU train batch size                                      |
|                                | `per_device_eval_batch_size`  | `cfg.eval_batch_size`             | *(from YAML/override)*         | Per-GPU eval batch size                                       |
| **Precision**                  | `fp16`                        | `cfg.fp16`                        | `False`                        | Mixed precision (if GPU supports)                             |
| **Evaluation & Checkpointing** | `evaluation_strategy`         | `cfg.evaluation_strategy`         | *(from YAML)*                  | e.g., `steps` or `epoch`                                      |
|                                | `eval_steps`                  | `cfg.eval_steps`                  | *(from YAML)*                  | Step interval for eval                                        |
|                                | `save_strategy`               | fixed                             | `steps`                        | Always save by steps                                          |
|                                | `save_steps`                  | `cfg.save_steps`                  | *(from YAML)*                  | Step interval for checkpoints                                 |
|                                | `save_total_limit`            | `cfg.save_total_limit`            | *(from YAML)*                  | Keep last N checkpoints                                       |
|                                | `load_best_model_at_end`      | `cfg.load_best_model_at_end`      | *(from YAML)*                  | Restore best checkpoint                                       |
|                                | `metric_for_best_model`       | `cfg.metric_for_best_model`       | *(from YAML)*                  | e.g., `eval_f1_weighted`                                      |
|                                | `greater_is_better`           | `cfg.greater_is_better`           | *(from YAML)*                  | True/False based on metric                                    |
|                                | `EarlyStoppingCallback`       | fixed                             | `patience=5`                   | Stop if no improvement                                        |
| **Logging**                    | `logging_steps`               | `cfg.logging_steps`               | *(from YAML)*                  | TB log frequency                                              |
|                                | `report_to`                   | fixed                             | `["tensorboard"]`              | Logging backend                                               |
|                                | `logging_dir`                 | fixed                             | `logs/training-logs`           | TensorBoard log dir                                           |
| **Dataloader**                 | `dataloader_num_workers`      | OS-aware                          | `0` on Windows, else `4`       | Worker threads per DataLoader                                 |
|                                | `dataloader_pin_memory`       | fixed                             | `False`                        | Pin memory disabled                                           |
| **Collation**                  | `data_collator`               | built                             | —                              | From `build_data_collator(fe)`                                |
| **Metrics**                    | `accuracy`                    | evaluate                          | —                              | `accuracy`                                                    |
|                                | `f1_weighted`                 | evaluate                          | —                              | `f1(average="weighted")`                                      |
|                                | `precision_weighted`          | evaluate                          | —                              | `precision(average="weighted")`                               |
|                                | `recall_weighted`             | evaluate                          | —                              | `recall(average="weighted")`                                  |
| **Push to Hub**                | `push_to_hub`                 | `cfg.push_to_hub`                 | `False`                        | HF Hub integration toggle                                     |


## 🖥️ Training Hardware & Environment

- **Device:** Laptop (Windows, WDDM driver model)
- **GPU:** NVIDIA GeForce **RTX 3080 Ti Laptop GPU** (16 GB VRAM)
- **Driver:** **576.52**
- **CUDA (driver):** **12.9**
- **PyTorch:** **2.8.0+cu129**
- **CUDA available:** ✅

---

## 📊 Training Logs & Metrics

- **Total FLOPs (training):** `7,703,275,221,221,900,000`
- **Training runtime:** `3,446.3047` seconds
- **Logging:** TensorBoard-compatible logs in `src/logs/training-logs/`

You can monitor training live with:

```bash
tensorboard --logdir src/logs/training-logs
```

### 📉 Loss Curve

The following plot shows the training loss progression:

![Training Loss Curve](assets/train_loss.svg)

*(SVG file generated during training and stored under `assets/`)*
## Inference 
```bash
python -m src.infer \
  --model_dir ./checkpoints/kws_w2v2 \
  --wav_path /path/to/your.wav \
  --top_k 5
```

## Evaluate on Test Set
```bash
python -m src.evaluate_fn --model_dir ./checkpoints/kws_w2v2
```


## AWS SageMaker — Train, Deploy, Batch

> This repository includes a minimal but complete SageMaker setup (scripts + client + CI).

### Prerequisites
- AWS account + `sagemaker` and `ecr` permissions.
- Set `AWS_REGION` and (if using GitHub Actions) `AWS_ROLE_TO_ASSUME` as repository variables/secrets.

### Quick commands (via `Makefile`)
```bash
# 1) Launch a training job
make train

# 2) Deploy a realtime endpoint
make deploy
# or deploy Serverless Inference
make deploy-sls

# 3) Run a Batch Transform job over a JSONL manifest in S3
make batch

# 4) Tear down the endpoint
make delete
```

### What the scripts do
- **`sagemaker/launch_training.py`** – spins up a Hugging Face training job. Region defaults to your session (`boto3.Session().region_name` or `us-east-1`). 
- **`sagemaker/deploy_realtime.py`** – creates a Hugging Face model and deploys a realtime endpoint; supports **Serverless Inference** when `SERVERLESS=true`.
- **`sagemaker/batch_transform.py`** – runs offline inference using a JSONL manifest in S3. Set env vars (examples):  
  `MODEL_S3` (model tarball), `INPUT_JSONL_S3` (JSONL path), `OUTPUT_S3` (optional), `BT_INSTANCES`, `BT_INSTANCE_TYPE`.  
  Each JSONL line looks like:  
  ```json
  {"inputs": {"s3_uri": "s3://bucket/key.wav"}, "parameters": {"top_k": 5}}
  ```
- **`sagemaker/code/inference.py`** – custom entrypoint for Hugging Face Inference Toolkit. **Accepted inputs** (any of):  
  - `{"inputs": {"base64": "<...>"}}` (WAV bytes)  
  - `{"inputs": {"s3_uri": "s3://..."}}`  
  - `{"inputs": {"url": "https://..."}}`  
  - `{"inputs": {"array": [...], "sampling_rate": 16000}}`  
  Returns top‑K labels/scores.
- **`client/invoke_realtime.py`** – tiny invoker:  
  ```bash
  AWS_REGION=us-east-1 ENDPOINT_NAME=kws-realtime WAV_PATH=sample.wav TOP_K=5   python client/invoke_realtime.py
  ```

### CI (GitHub Actions)
- Workflow: `.github/workflows/ci.yml` → checkout → set up Python 3.10 → install `requirements.txt` → run `ruff` and `pytest`.  
  On `main`, if `AWS_ROLE_TO_ASSUME` is set, it **configures AWS creds** and runs `python sagemaker/pipeline.py`.

---

## Tips & Troubleshooting
- Use `-h` on any script (e.g., `python -m src.train -h`) to see all flags.
- If you previously saw `requireeements.txt`, note it’s been **renamed** to `requirements.txt`.
- For realtime inference audio I/O issues, check microphone permissions and default input device.
- If CUDA mismatches occur, verify your driver/runtime pairing (example run used CUDA 12.9 with PyTorch 2.8.0+cu129).

---

## Roadmap
- Confusion matrix and per‑class metrics visualization.
- More keyword sets and multilingual support.
- Quantization / distillation + mobile demo (TFLite/CoreML).

## Acknowledgements
- Hugging Face `transformers` and `datasets`
- Google **Speech Commands** dataset