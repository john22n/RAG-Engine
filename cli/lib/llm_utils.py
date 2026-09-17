import os
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from .prompts.spell import expand_prompt, rewrite_prompt, spell_prompt

load_dotenv()

class LLM:
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-40"

    def __init__(self, api_key: str | None = None, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL):
        self.api_key = os.environ.get("OPEN_ROUTER_KEY")
        if not self.api_key:
            raise ValueError(" api key not provieded")

        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model

    def enhance_query(self, query:str, enhance: Literal['spell', 'rewrite', 'expand'] | None = None) -> str | None:
        prompt = spell_prompt

        match enhance:
            case 'spell':
                prompt = spell_prompt
            case 'rewrite':
                promtp = rewrite_prompt
            case 'expand':
                prompt = expand_prompt

        enhanced = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt(query)
                        }
                    ]
                )

        enhanced_query = enhanced.choices[0].message.content
        return enhanced_query


