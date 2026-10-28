from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — OpenRouter (OpenAI-compatible)
    openrouter_api_key: str
    llm_model: str = "openai/gpt-oss-120b"
    # Enrutamiento de proveedor fijado en cada request (no solo en la cuenta de
    # OpenRouter): solo este proveedor, sin fallback a otros, y excluyendo
    # proveedores que guardan datos. Vacío = sin restricción.
    llm_provider_only: str = "groq"
    llm_allow_fallbacks: bool = False
    llm_data_collection: str = "deny"

    # MongoDB Atlas
    mongodb_uri: str
    mongodb_db_name: str = "ampara"
    mongodb_collection_chunks: str = "chunks_collection"

    # Model Service (ONNX) — usar "mock" para desarrollo
    model_service_url: str = "mock"
    embedding_endpoint: str = "/embed"
    embedding_base_endpoint: str = "/embed_base"
    # Reranker: NO se usa en el pipeline de producción (app/graph/rag_graph.py
    # solo recupera con el embedding finetuneado — decisión tras el
    # experimento de ablación). Sigue existiendo porque scripts/eval_ablation.py
    # (config hybrid_rerank) todavía lo necesita como punto de comparación.
    reranker_endpoint: str = "/rerank"

    # OpenRouter base URL (compartido por el LLM del grafo y el judge de RAGAS)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # RAGAS — modelos juez
    ragas_llm_model: str = "deepseek/deepseek-v4-flash"
    # Embeddings para el script de evaluación batch (HuggingFace local)
    ragas_embeddings_model: str = "nomic-ai/nomic-embed-text-v1.5"
    # Embeddings para la evaluación en tiempo real (API via OpenRouter)
    ragas_realtime_embeddings_model: str = "qwen/qwen3-embedding-8b"

    # PostgreSQL (métricas RAGAS y admin — vacío = deshabilitado)
    postgres_dsn: str = ""

    # RAGAS evaluación por consulta individual (requiere postgres_dsn configurado)
    ragas_enabled: bool = False

    # JWT autenticación admin
    jwt_secret_key: str = "change-me-in-production"
    jwt_expire_minutes: int = 480
    # Bootstrap del primer admin al arrancar (app/main.py lifespan) — vacío =
    # deshabilitado. Idempotente: solo crea si no existe ningún admin todavía
    # (mismo guard que scripts/create_first_admin.py, que sigue disponible
    # para crear admins manualmente sin tocar el .env).
    admin_username: str = ""
    admin_password: str = ""

    # App
    app_env: str = "development"
    session_max_messages: int = 5
    # Historial de una sesión inactiva más de este tiempo se borra de memoria
    # (InMemorySaver no expira nada por sí solo).
    session_idle_ttl_minutes: int = 15
    # Umbral del reranker — NO se usa en producción (ver nota de
    # reranker_endpoint arriba), solo lo lee scripts/eval_ablation.py.
    reranker_score_threshold: float = 0.5

    # Protección de costos para la validación cualitativa (usuarios reales
    # probando el chatbot): tope de consultas por IP, largo máximo del
    # mensaje de entrada (tokens reales, cl100k_base — mismo encoding que
    # app/services/document_processor.py), y tope de tokens de salida del
    # LLM generador.
    max_queries_per_ip: int = 10
    max_message_tokens: int = 4096
    llm_max_output_tokens: int = 8192


settings = Settings()
