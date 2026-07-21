from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://Adil-Amjad.services.ai.azure.com/openai/v1"
deployment_name = "Phi-4-mini-instruct"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key="YOUR_API_KEY",
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is chatgpt?",
        }
    ],
)

print(completion.choices[0].message)