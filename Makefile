.PHONY: train deploy deploy-sls batch delete

train:
	python sagemaker/launch_training.py

deploy:
	python sagemaker/deploy_realtime.py

deploy-sls:
	SERVERLESS=true python sagemaker/deploy_realtime.py

batch:
	python sagemaker/batch_transform.py

delete:
	python - <<'PY'
import os, boto3
sm = boto3.client('sagemaker')
name = os.getenv('ENDPOINT_NAME','kws-realtime')
try: sm.delete_endpoint(EndpointName=name)
except sm.exceptions.ClientError as e: print(e)
PY
