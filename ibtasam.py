from openai import OpenAI
import os
import os
from dotenv import load_dotenv

load_dotenv()

# Azure AI Foundry endpoint
endpoint = "https://ctp09.services.ai.azure.com/openai/v1"

# Your deployed model name (must exactly match the deployment name in Azure AI Foundry)
deployment_name = "Phi-4-mini-instruct"

# Create the client
client = OpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=endpoint
)

# Send a request
response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "tell me about islamabad"
        }
    ]
)

# Print only the generated text
print(response.choices[0].message.content)