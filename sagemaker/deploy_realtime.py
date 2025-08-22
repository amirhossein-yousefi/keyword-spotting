import os
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

sess = sagemaker.Session()
role = os.getenv("SM_EXECUTION_ROLE_ARN") or sagemaker.get_execution_role()
model_data = os.environ["MODEL_S3"]  # s3://.../model.tar.gz from training

env = {
    # task = "audio-classification" for pipelines; we still override with our handler
    "HF_TASK": "audio-classification"
}

# Attach user-defined inference handler via entry_point/source_dir.
model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version=os.getenv("TRANSFORMERS_VERSION", "4.26"),
    pytorch_version=os.getenv("PYTORCH_VERSION", "1.13"),
    py_version=os.getenv("PY_VERSION", "py39"),
    env=env,
    entry_point="inference.py",
    source_dir="sagemaker/code",
)

endpoint_name = os.getenv("ENDPOINT_NAME", "kws-realtime")

if os.getenv("SERVERLESS", "false").lower() == "true":
    # Serverless inference (great for spiky/low-traffic audio KWS)
    # https://huggingface.co/docs/sagemaker/en/inference
    predictor = model.deploy(
        serverless_inference_config=ServerlessInferenceConfig(
            memory_size_in_mb=int(os.getenv("SERVERLESS_MEM_MB", "4096")),
            max_concurrency=int(os.getenv("SERVERLESS_CONCURRENCY", "5")),
        ),
        endpoint_name=endpoint_name,
    )
else:
    predictor = model.deploy(
        initial_instance_count=int(os.getenv("NUM_INSTANCES", "1")),
        instance_type=os.getenv("INSTANCE_TYPE", "ml.c6i.xlarge"),
        endpoint_name=endpoint_name,
    )

print("Endpoint:", predictor.endpoint_name)
