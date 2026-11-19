import os
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env", override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

print("groq", version("groq"))

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Reply with the single word: ready",
        }
    ],
)

print(response.choices[0].message.content)