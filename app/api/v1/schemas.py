import tiktoken
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

_token_encoder = tiktoken.get_encoding("cl100k_base")


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

    @field_validator("message")
    @classmethod
    def _limit_message_tokens(cls, value: str) -> str:
        # Cuenta tokens reales (no caracteres) — bloquea mensajes larguísimos
        # (ej. prompt injection con contexto inflado) desde la validación de
        # Pydantic, antes de que toquen el grafo RAG o gasten un solo token
        # de LLM real.
        n_tokens = len(_token_encoder.encode(value))
        if n_tokens > settings.max_message_tokens:
            raise ValueError(
                f"El mensaje tiene {n_tokens} tokens, el máximo permitido es "
                f"{settings.max_message_tokens}."
            )
        return value


class SourceDocument(BaseModel):
    source_type: str   # "Ley 30364" | "Código Penal" | "Jurisprudencia · Casación" | ...
    source_label: str  # título del artículo o referencia jurisprudencial
    excerpt: str       # primeros ~250 chars del texto del chunk
    score: float       # score del embedding (vectorSearchScore), útil para debug/tesis


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDocument] = []
    from_context: bool = False  # True cuando se respondió desde el historial
    out_of_domain: bool = False
    no_results: bool = False
    consulta_id: int | None = None
