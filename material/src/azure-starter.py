

from google.colab import userdata
token=userdata.get('GITHUB_TOKEN')

###
#!pip install azure-ai-inference
###

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential


endpoint = "https://models.inference.ai.azure.com"
model_name = "Phi-3.5-MoE-instruct"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

messages = [
    SystemMessage("You are a helpful assistant."),
    UserMessage("What is the capital of France?"),
    AssistantMessage("The capital of France is Paris."),
    UserMessage("And what about Germany?")
]

response = client.complete(
    messages=messages,
    model=model_name,
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000
)

response.choices[0].message.content


