# One-off or scheduled offline inference over large audio sets.
# Uses JSONL with each line like:
#   {"inputs":{"s3_uri":"s3://bucket/path/file.wav"},"parameters":{"top_k":5}}
#
# HF DLC supports .jsonl for batch transform.
# https://huggingface.co/docs/sagemaker/en/inference

import os, sagemaker
from sagemaker.huggingface import HuggingFaceModel

sess = sagemaker.Session()
role = os.getenv("SM_EXECUTION_ROLE_ARN") or sagemaker.get_execution_role()
model_data = os.environ["MODEL_S3"]

model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version=os.getenv("TRANSFORMERS_VERSION", "4.26"),
    pytorch_version=os.getenv("PYTORCH_VERSION", "1.13"),
    py_version=os.getenv("PY_VERSION", "py39"),
    entry_point="inference.py",
    source_dir="sagemaker/code",
    env={"HF_TASK": "audio-classification"},
)

transformer = model.transformer(
    instance_count=int(os.getenv("BT_INSTANCES", "1")),
    instance_type=os.getenv("BT_INSTANCE_TYPE", "ml.c6i.2xlarge"),
    strategy="SingleRecord",
    assemble_with="Line",
    output_path=os.getenv("OUTPUT_S3", f"s3://{sess.default_bucket()}/kws/batch-output/"),
)

transformer.transform(
    data=os.environ["INPUT_JSONL_S3"],      # s3://bucket/prefix/input.jsonl
    content_type="application/json",
    split_type="Line",
    wait=True,
)
print("Batch outputs:", os.getenv("OUTPUT_S3"))
