from openai import OpenAI
import os

endpoint = "https://abdulah.services.ai.azure.com/openai/v1/"
model_name = "Phi-4-mini-instruct"
deployment_name = "Phi-4-mini-instruct"

api_key = os.getenv("API_KEY")

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)