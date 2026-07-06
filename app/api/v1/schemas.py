from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class SourceDocument(BaseModel):
    source_type: str   # "Ley 30364" | "Código Penal" | "Jurisprudencia · Casación" | ...
    source_label: str  # título del artículo o referencia jurisprudencial
    excerpt: str       # primeros ~250 chars del texto del chunk
    score: float       # rerank score (útil para debug/tesis)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDocument] = []
    from_context: bool = False  # True cuando se respondió desde el historial
    out_of_domain: bool = False
    no_results: bool = False
    consulta_id: int | None = None
