# hf-kws — Keyword Spotting with Hugging Face + PyTorch
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
└── configs/
    └── train_config.yaml
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