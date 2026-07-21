import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class LLMConfig:

    provider: str = "groq"

    model_name: str = "llama-3.3-70b-versatile"

    base_url: str | None = None

    api_key: str | None = os.getenv(
        "GROQ_API_KEY"
    )

    temperature: float = 0.1

    max_tokens: int = 512

    timeout: int = 120