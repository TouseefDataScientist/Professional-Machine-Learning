import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the secret key from the .env file
load_dotenv()

# Setup Azure endpoint details
endpoint = "https://uzair-foundry-test.services.ai.azure.com/openai/v1/"
deployment_name = "Phi-4-reasoning"

# Fetch the key securely
api_key = os.getenv("AZURE_API_KEY")

# Initialize the client
client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

# Call the Phi-4 model
completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

# Print the clean response text
print(completion.choices[0].message.content)