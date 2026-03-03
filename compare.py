import os
import sys
import json
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from travel_brain import config
from google import genai

# A highly specific question that tests recent/niche knowledge
QUERY = "What exactly do travel vloggers say about Bias Tugal Beach in Bali? Explain the path to get there and any animals they say to watch out for."

def ask_base_model():
    print("="*60)
    print("🤖 BASE LLM ONLY (gemma-3-27b-it NO RAG)")
    print("="*60)
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemma-3-27b-it',
            contents=QUERY,
        )
        print(response.text)
    except Exception as e:
        print(f"Error calling base model: {e}")
    print("\n")

def ask_rag_model():
    print("="*60)
    print("🧠 TRAVEL BRAIN (gemma-3-27b-it + 185 RAG Chunks)")
    print("="*60)
    try:
        res = requests.post(
            "http://localhost:8000/chat/stream",
            json={"message": QUERY, "history": [], "top_k": 5},
            stream=True
        )
        for line in res.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data = json.loads(decoded[6:])
                    if data.get("type") == "delta":
                        # Print without newline to simulate streaming
                        print(data.get("content", ""), end="")
        print()
    except Exception as e:
        print(f"Error calling RAG API: {e}")
    print("\n")

if __name__ == "__main__":
    print(f"\n❓ QUESTION: {QUERY}\n")
    ask_base_model()
    ask_rag_model()
