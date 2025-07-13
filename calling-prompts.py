import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Create the LLM object
llm = ChatOpenAI(temperature=0.9)

# Prompt
prompt = input("Ask questions")

# Generate response
response = llm.invoke(prompt)

# Print result
print("Suggested Title:", response.content)