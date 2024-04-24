import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import time

boto3.setup_default_session(profile_name='analitiq')
region = 'eu-central-1'

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20231212T174551')['Role']['Arn']

image_uri = get_huggingface_llm_image_uri(
    backend="huggingface", # or lmi
    region=region
)

model_name = "falcon-7b-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

hub = {
    'HF_MODEL_ID':'tiiuae/falcon-7b',
    'HF_TASK':'text-generation',
    'SM_NUM_GPUS':'1' # Setting to 1 because sharding is not supported for this model
}

model = HuggingFaceModel(
    name=model_name,
    env=hub,
    role=role,
    image_uri=image_uri
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name=model_name
)

