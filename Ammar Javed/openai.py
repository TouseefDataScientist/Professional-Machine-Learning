from openai import OpenAI

endpoint = "https://AmmarCTP.services.ai.azure.com/openai/v1"
deployment_name = "Phi-4"
api_key = "Your_Api_key_hear"

client = OpenAI(
    base_url=endpoint,
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