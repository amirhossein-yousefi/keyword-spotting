import base64, json, os, boto3

smr = boto3.client("sagemaker-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
endpoint = os.getenv("ENDPOINT_NAME", "kws-realtime")

with open(os.getenv("WAV_PATH", "sample.wav"), "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "inputs": {"base64": b64},
    "parameters": {"top_k": int(os.getenv("TOP_K", "5"))}
}
resp = smr.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(resp["Body"].read().decode("utf-8"))
