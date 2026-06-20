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
    embedding_base_endpoint: str = "/embed_base"
    reranker_endpoint: str = "/rerank"

    # OpenRouter base URL (compartido por el LLM del grafo y el judge de RAGAS)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # RAGAS — modelos juez
    ragas_llm_model: str = "deepseek/deepseek-v4-flash"
    ragas_embeddings_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # PostgreSQL (métricas RAGAS — vacío = deshabilitado)
    postgres_dsn: str = ""

    # RAGAS evaluación por consulta individual (requiere postgres_dsn configurado)
    ragas_enabled: bool = False

    # App
    app_env: str = "development"
    session_max_messages: int = 5
    reranker_score_threshold: float = 0.5


settings = Settings()
