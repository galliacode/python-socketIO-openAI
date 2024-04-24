
import json
import boto3


endpoint_name = "analitiq"


def query_endpoint(payload):
    client = boto3.client("runtime.sagemaker",
                          region_name='eu-central-1',
                          aws_access_key_id='',
                          aws_secret_access_key='',
                          aws_session_token='',
                          )
    response = client.invoke_endpoint(
        EndpointName=endpoint_name
        , InferenceComponentName='jumpstart-dft-hf-llm-mistral-7b-ins-20240110-1-20240110-1632180'
        , ContentType="application/json", Body=json.dumps(payload).encode("utf-8")
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response


from typing import Dict, List


def format_instructions(instructions: List[Dict[str, str]]) -> List[str]:
    """Format instructions where conversation roles must alternate user/assistant/user/assistant/..."""
    prompt: List[str] = []
    for user, answer in zip(instructions[::2], instructions[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (instructions[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)


def print_instructions(prompt: str, response: str) -> None:
    bold, unbold = '\033[1m', '\033[0m'
    print(f"{bold}> Input{unbold}\n{prompt}\n\n{bold}> Output{unbold}\n{response[0]['generated_text']}\n")


instructions = [{"role": "user", "content": "what is the recipe of mayonnaise?"}]
prompt = format_instructions(instructions)
payload = {
    "inputs": prompt,
    "parameters": {"max_new_tokens": 256, "do_sample": True}
}
response = query_endpoint(payload)
print_instructions(prompt, response)


instructions = [
    {"role": "user", "content": "I am going to Paris, what should I see?"},
    {
        "role": "assistant",
        "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
    },
    {"role": "user", "content": "What is so great about #1?"},
]
prompt = format_instructions(instructions)
payload = {
    "inputs": prompt,
    "parameters": {"max_new_tokens": 256, "do_sample": True}
}
response = query_endpoint(payload)
print_instructions(prompt, response)


instructions = [
    {
        "role": "user",
        "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
    }
]
prompt = format_instructions(instructions)
payload = {
    "inputs": prompt,
    "parameters": {"max_new_tokens": 256, "do_sample": True, "temperature": 0.2}
}
response = query_endpoint(payload)
print_instructions(prompt, response)


instructions = [
    {
        "role": "user",
        "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
    }
]
prompt = format_instructions(instructions)
payload = {
    "inputs": prompt,
    "parameters": {"max_new_tokens": 256, "do_sample": True, "temperature": 0.2}
}
response = query_endpoint(payload)
print_instructions(prompt, response)
