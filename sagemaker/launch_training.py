import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

region = os.getenv("AWS_REGION", boto3.Session().region_name or "us-east-1")
sess = sagemaker.Session()
role = os.getenv("SM_EXECUTION_ROLE_ARN") or sagemaker.get_execution_role()

# Where to checkpoint (spot-friendly) and where to stage data (optional)
bucket = os.getenv("SAGEMAKER_BUCKET", f"sagemaker-{region}-{sess.account_id()}")
chkpt_s3 = f"s3://{bucket}/kws/checkpoints/"
job_name_prefix = os.getenv("JOB_NAME_PREFIX", "kws-wav2vec2")

# Hyperparameters passed as CLI args to src/train.py (see your README)
hyperparameters = {
    "checkpoint": os.getenv("HF_CHECKPOINT", "facebook/wav2vec2-base"),
    "output_dir": "/opt/ml/model",           # SageMaker will tarball & upload
    "num_train_epochs": int(os.getenv("NUM_EPOCHS", "8")),
    "per_device_train_batch_size": int(os.getenv("TRAIN_BS", "16")),
    "per_device_eval_batch_size": int(os.getenv("EVAL_BS", "16")),
    # add any of your script's flags here (e.g., lr, warmup_steps, etc.)
}

# Pick a HF DLC version that exists in your region; see "Available DLCs on AWS".
# You can adjust these three to match your local libs if needed.
# https://huggingface.co/docs/sagemaker/en/train  (Estimator params)
estimator = HuggingFace(
    entry_point="src/train.py",
    source_dir=".",                          # send the whole repo (small)
    role=role,
    instance_type=os.getenv("SM_TRAIN_INSTANCE", "ml.g5.xlarge"),
    instance_count=int(os.getenv("SM_TRAIN_COUNT", "1")),
    transformers_version=os.getenv("TRANSFORMERS_VERSION", "4.26"),
    pytorch_version=os.getenv("PYTORCH_VERSION", "1.13"),
    py_version=os.getenv("PY_VERSION", "py39"),
    hyperparameters=hyperparameters,
    checkpoint_s3_uri=chkpt_s3,
    output_path=f"s3://{bucket}/kws/output/",
    enable_sagemaker_metrics=True,
    use_spot_instances=True,
    max_run=int(os.getenv("MAX_RUN", "43200")),   # 12h
    max_wait=int(os.getenv("MAX_WAIT", "43200")), # Spot window
    volume_size=int(os.getenv("VOLUME_SIZE", "100")),
    environment={"WANDB_DISABLED": "true"},       # example: quiet containers
)

# If your script downloads HF datasets itself, no channels are required:
# estimator.fit()
# Otherwise, you can pass S3 channels: {'train': 's3://...', 'test':'s3://...'}.
estimator.fit(wait=True, job_name=os.getenv("JOB_NAME"))

print("Model artifacts:", estimator.model_data)  # s3://.../model.tar.gz
