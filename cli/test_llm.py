import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.environ.get("OPEN_ROUTER_KEY")
if not api_key:
    raise RuntimeError("open router key env not imported")

client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        )
