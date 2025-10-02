import os
import sys
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv(".env")

HF_TOKEN = os.getenv("API_KEY")
if not HF_TOKEN:
    print("Set HF_TOKEN environment variable first.")
    sys.exit(1)

client = InferenceClient(token=HF_TOKEN)
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def ask(prompt: str) -> str:
    """Send prompt to Hugging Face LLaMA model and return response."""
    resp = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message["content"].strip()

def main():
    print(f"Hugging Face Q&A CLI — using model: {MODEL}")
    print("Type a question and press Enter (Ctrl-C to quit).")
    try:
        while True:
            prompt = input("\nYou: ").strip()
            if not prompt:
                continue
            print("Thinking...\n")
            answer = ask(prompt)
            print("Bot:", answer)
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
