import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Find the repository root and its .env file
project_root = Path(__file__).resolve().parents[2]
print(f"Loading GROQ_API_KEY from {project_root / '.env'}")
env_path = project_root / ".env"

# Load GROQ_API_KEY from the root .env
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(f"GROQ_API_KEY was not found in {env_path}")

# Groq reads GROQ_API_KEY from the environment
client = Groq()

# Sort the IDs to produce stable output
model_ids = sorted(model.id for model in client.models.list().data)

for model_id in model_ids:
    print(model_id)


models = {model.id: model for model in client.models.list().data}

for name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    model = models[name]
    print(name, "| made by", model.owned_by, "| weights:", model.hugging_face_id)


# Asking the 20b and the 120b the same question
print("\n")
question = [{"role": "user", "content": "In one sentence, what is a refund?"}]

for name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    response = client.chat.completions.create(model=name, messages=question)
    print(name)
    print("  ", response.choices[0].message.content)


# Counting the tokens
import tiktoken

encoding = tiktoken.get_encoding("o200k_harmony")  # the tokenizer gpt-oss uses

ids = encoding.encode("hi how are you?")
print("Encoding for 'hi how are you?':", ids)
print([encoding.decode([i]) for i in ids])
print(len(ids), "tokens")

# Splitting rare words and other languages
print("\n")
print("Splitting rare words and other languages:")

for text in ["refund", "unsubscribing", "My parcel has not arrived", "मेरा पार्सल नहीं आया"]:
    ids = encoding.encode(text)
    print(len(ids), [encoding.decode([i]) for i in ids])

print("\n")
print("Decoding ids back to text:  ")
ids = encoding.encode("The parcel arrived but the box was crushed")
print(ids)
print(encoding.decode(ids))
print(encoding.n_vocab, "tokens in the vocabulary")

# Reading Groq's prices
print("\n")
print(" Groq's prices for the 20b and 120b models:")

for model in client.models.list().data:
    if model.id in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        per_million_in = float(model.pricing["prompt"]) * 1_000_000
        per_million_out = float(model.pricing["completion"]) * 1_000_000
        print(f"{model.id}: ${per_million_in:.3f} in, ${per_million_out:.2f} out, per million tokens")


# Pricing one ticket
print("\n")
print("Pricing one ticket:")

prices = next(model.pricing for model in client.models.list().data if model.id == "openai/gpt-oss-120b")

messages = [{"role": "system", "content": structured}, *examples, {"role": "user", "content": "My parcel has not arrived"}]
response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=messages)
usage = response.usage

cost = usage.prompt_tokens * float(prices["prompt"]) + usage.completion_tokens * float(prices["completion"])
print(usage.prompt_tokens, "tokens in,", usage.completion_tokens, "tokens out")
print(f"${cost:.6f} for this ticket")
print(f"${cost * 50_000:.2f} for 50,000 tickets")