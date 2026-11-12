from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "VectorPulse Analytics Engine"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300
    max_matrix_rows: int = 500_000

    class Config:
        env_prefix = "VECTORPULSE_"


settings = Settings()
