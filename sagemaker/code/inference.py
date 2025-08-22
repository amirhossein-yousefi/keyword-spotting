# Custom user-defined inference for the Hugging Face Inference Toolkit.
# Supports inputs:
#   {"inputs": {"base64": "<...>"}}   # base64-encoded WAV
#   {"inputs": {"s3_uri": "s3://bucket/key.wav"}}
#   {"inputs": {"url": "https://.../file.wav"}}
#   {"inputs": {"array": [...], "sampling_rate": 16000}}  # already-decoded
#
# Returns top-k labels/scores per HF audio-classification pipeline.

import base64
import io
import json
from typing import Any, Dict, Tuple

import numpy as np
from transformers import pipeline

# optional imports used in input_fn; installed via code/requirements.txt
import boto3, requests, soundfile as sf  # type: ignore[import-not-found]


def _wav_to_array_and_sr(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:  # mixdown to mono
        audio = audio.mean(axis=1)
    return audio, int(sr)


def model_fn(model_dir: str):
    # The HF Inference Toolkit will call this once per worker to load the model.
    # We create a pipeline to keep the handler concise.
    clf = pipeline(
        task="audio-classification",  # HF_TASK equivalent
        model=model_dir,
        feature_extractor=model_dir,  # processor/feature extractor co-saved
        top_k=None,                   # we’ll control via request parameters
    )
    return clf


def input_fn(request_body: str, content_type: str = "application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)
    params = payload.get("parameters", {})
    x = payload.get("inputs", payload)

    # 1) base64-encoded wav
    if isinstance(x, dict) and "base64" in x:
        wav_bytes = base64.b64decode(x["base64"])
        audio, sr = _wav_to_array_and_sr(wav_bytes)
        return {"array": audio.tolist(), "sampling_rate": sr}, params

    # 2) s3 uri
    if isinstance(x, dict) and "s3_uri" in x:
        s3 = boto3.client("s3")
        uri = x["s3_uri"]
        if not uri.startswith("s3://"):
            raise ValueError("s3_uri must start with s3://")
        bucket_key = uri[len("s3://") :]
        bucket, key = bucket_key.split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        wav_bytes = obj["Body"].read()
        audio, sr = _wav_to_array_and_sr(wav_bytes)
        return {"array": audio.tolist(), "sampling_rate": sr}, params

    # 3) https url
    if isinstance(x, dict) and "url" in x:
        resp = requests.get(x["url"], timeout=10)
        resp.raise_for_status()
        audio, sr = _wav_to_array_and_sr(resp.content)
        return {"array": audio.tolist(), "sampling_rate": sr}, params

    # 4) pre-parsed array/sampling_rate or anything pipeline() already accepts
    return x, params


def predict_fn(data_and_params: Any, model):
    inputs, params = data_and_params
    # Pass through request-time parameters like top_k
    return model(inputs, **params)


def output_fn(prediction, accept: str = "application/json"):
    # HF toolkit expects the string body.
    return json.dumps(prediction)
