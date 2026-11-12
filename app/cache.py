import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as aioredis


class AnalyticsCache:
    """Cache layer backed by Redis with automatic in-memory fallback."""

    def __init__(self, redis_url: str, ttl_seconds: int = 300) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._redis: Optional[aioredis.Redis] = None
        self._redis_loop: Optional[asyncio.AbstractEventLoop] = None
        self._memory_store: Dict[str, Tuple[float, str]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def build_key(namespace: str, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"vp:{namespace}:{digest}"

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        loop = asyncio.get_running_loop()
        if self._redis is not None and self._redis_loop is not loop:
            self._redis = None
            self._redis_loop = None

        if self._redis is None:
            try:
                client = aioredis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
                await client.ping()
                self._redis = client
                self._redis_loop = loop
            except Exception:
                self._redis = None
                self._redis_loop = None
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        client = await self._get_redis()
        if client is not None:
            try:
                raw = await client.get(key)
                if raw is not None:
                    self.hits += 1
                    return json.loads(raw)
                self.misses += 1
                return None
            except Exception:
                pass

        entry = self._memory_store.get(key)
        if entry is not None:
            expires_at, raw = entry
            if time.monotonic() < expires_at:
                self.hits += 1
                return json.loads(raw)
            del self._memory_store[key]

        self.misses += 1
        return None

    async def set(self, key: str, value: Any) -> None:
        raw = json.dumps(value)
        client = await self._get_redis()
        if client is not None:
            try:
                await client.setex(key, self.ttl_seconds, raw)
                return
            except Exception:
                pass

        self._memory_store[key] = (time.monotonic() + self.ttl_seconds, raw)

    async def backend_status(self) -> str:
        client = await self._get_redis()
        return "redis" if client is not None else "in-memory-fallback"

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
