import hashlib
import json

import redis

from rag.cache.config import RedisConfig


class RedisCacheService:

    def __init__(
        self,
        config: RedisConfig,
    ):
        self.config = config
        self.client = None

        if not self.config.enabled:
            return

        if not self.config.url:
            print(
                "Redis disabled: "
                "REDIS_URL not configured."
            )
            return

        try:

            self.client = (
                redis.Redis.from_url(
                    self.config.url,
                    decode_responses=True,
                )
            )

            self.client.ping()

            print(
                "Redis connection successful."
            )

        except Exception as exc:

            print(
                "Redis connection failed:",
                exc,
            )

            self.client = None

    def build_key(
        self,
        tenant_id: int,
        departments: tuple[str, ...],
        query: str,
    ) -> str:

        normalized_query = (
            " ".join(
                query.lower().split()
            )
        )

        normalized_departments = (
            ",".join(
                sorted(departments)
            )
        )

        raw_key = (
            f"{tenant_id}:"
            f"{normalized_departments}:"
            f"{normalized_query}"
        )

        key_hash = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return (
            f"ekip:rag:{key_hash}"
        )

    def get(
        self,
        key: str,
    ) -> dict | None:

        if self.client is None:
            return None

        try:

            value = self.client.get(
                key
            )

            if value is None:
                return None

            return json.loads(
                value
            )

        except Exception as exc:

            print(
                "Redis GET failed:",
                exc,
            )

            return None

    def set(
        self,
        key: str,
        value: dict,
    ) -> bool:

        if self.client is None:
            return False

        try:

            self.client.setex(
                key,
                self.config.ttl_seconds,
                json.dumps(value),
            )

            return True

        except Exception as exc:

            print(
                "Redis SET failed:",
                exc,
            )

            return False
        
    def get_response(
        self,
        cache_id: str,
    ) -> dict | None:

        return self.get(
            f"ekip:semantic:{cache_id}"
        )


    def set_response(
        self,
        cache_id: str,
        value: dict,
    ) -> bool:

        if self.client is None:
            return False

        try:

            self.client.setex(
                f"ekip:semantic:{cache_id}",
                self.config.ttl_seconds,
                json.dumps(value),
            )

            return True

        except Exception as exc:

            print(
                "Redis semantic SET failed:",
                exc,
            )

            return False