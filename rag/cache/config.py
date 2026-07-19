import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class RedisConfig:

    url: str | None = os.getenv(
        "REDIS_URL"
    )

    enabled: bool = True

    ttl_seconds: int = 3600