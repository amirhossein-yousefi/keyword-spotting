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