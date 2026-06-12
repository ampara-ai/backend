from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — OpenRouter (OpenAI-compatible)
    openrouter_api_key: str
    llm_model: str = "openai/gpt-oss-120b"

    # MongoDB Atlas
    mongodb_uri: str
    mongodb_db_name: str = "ampara"
    mongodb_collection_chunks: str = "chunks_collection"

    # Model Service (ONNX) — usar "mock" para desarrollo
    model_service_url: str = "mock"
    embedding_endpoint: str = "/embed"
    reranker_endpoint: str = "/rerank"

    # PostgreSQL (métricas RAGAS — vacío = deshabilitado)
    postgres_dsn: str = ""

    # App
    app_env: str = "development"
    session_max_messages: int = 5
    reranker_score_threshold: float = 0.5


settings = Settings()
