import os
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(".env")

async def test():
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        config = types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.7
        )
        
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents="What is the stock price of AAPL right literally this second?",
                config=config
            )
            print("2.5 SUCCESS:")
            print("TEXT:", response.text[:200])
            if response.candidates and response.candidates[0].grounding_metadata:
                print("GROUNDING METADATA EXISTS")
        except Exception as e:
            print("2.5 ERROR:", e)
    except Exception as e:
        print("INIT ERROR: ", e)

if __name__ == "__main__":
    asyncio.run(test())
