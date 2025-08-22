# A concise SageMaker Pipelines example:
# - TrainingStep (your src/train.py via HuggingFace Estimator)
# - Model metrics passed to a ConditionStep
# - RegisterModel to the SageMaker Model Registry if metrics pass
#
# References: SageMaker Pipelines + HF examples
# https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html
# https://www.philschmid.de/mlops-sagemaker-huggingface-transformers

import os
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, CreateModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import RegisterModel
from sagemaker.huggingface import HuggingFace
from sagemaker import Session
from sagemaker.workflow.step_collections import TrainingStep

sess = Session()
role = os.getenv("SM_EXECUTION_ROLE_ARN")

p_instance_type = ParameterString("TrainInstanceType", default_value="ml.g5.xlarge")
p_epochs = ParameterInteger("Epochs", default_value=8)
p_acc_threshold = ParameterFloat("AccThreshold", default_value=0.90)

estimator = HuggingFace(
    entry_point="src/train.py",
    source_dir=".",
    role=role,
    instance_type=p_instance_type,
    instance_count=1,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    hyperparameters={
        "checkpoint": "facebook/wav2vec2-base",
        "num_train_epochs": p_epochs,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "output_dir": "/opt/ml/model",
    },
    enable_sagemaker_metrics=True,
)

# Training step
train_step = TrainingStep(
    name="TrainKWS",
    estimator=estimator,
    inputs={},  # your script pulls HF datasets itself
    cache_config=CacheConfig(enable_caching=True, expire_after="30d"),
)

# Example: assume your script writes eval metrics to evaluation.json in /opt/ml/model
# and you save it into model.tar.gz; we attach a PropertyFile to read "accuracy".
eval_report = PropertyFile(
    name="EvalReport",
    output_name="metrics",
    path="evaluation.json",   # ensure your training script writes this file
)

# Register the model if accuracy >= threshold
register_step = RegisterModel(
    name="RegisterKWSModel",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.c6i.xlarge"],
    transform_instances=["ml.c6i.xlarge"],
    model_package_group_name=os.getenv("MODEL_PKG_GROUP", "kws-models"),
)

cond = ConditionStep(
    name="AccuracyGate",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=train_step.properties.FinalMetricDataList[0]["Value"],  # or use eval_report once produced
            right=p_acc_threshold,
        )
    ],
    if_steps=[register_step],
    else_steps=[],
)

pipeline = Pipeline(
    name=os.getenv("SM_PIPELINE_NAME", "KWS-HF-Pipeline"),
    parameters=[p_instance_type, p_epochs, p_acc_threshold],
    steps=[train_step, cond],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    definition = pipeline.definition()
    print(definition)
