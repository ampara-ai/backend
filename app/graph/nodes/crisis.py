from pydantic import BaseModel

from app.core.config import settings
from app.core.dependencies import get_llm_structured
from app.core.prompt_provider import prompts
from app.graph.state import RAGState

# Textos fijos (no generados): se pueden auditar y no dependen del LLM. Mismos canales que prompts/generation.md.
SAFETY_NOTICE = (
    "Si estás en peligro ahora mismo, llama al 105 (Policía Nacional) y, si puedes, ve a un lugar seguro "
    "o busca a alguien de confianza. También puedes llamar gratis a la Línea 100, que brinda apoyo y "
    "orientación ante la violencia. Si la persona en peligro está fuera del Perú, llama al número de emergencias "
    "de ese país."
)
CRISIS_FOLLOW_UP = "Cuando estés a salvo, puedo explicarte cómo pedir medidas de protección o cómo denunciar."
CRISIS_ANSWER = f"{SAFETY_NOTICE}\n\n{CRISIS_FOLLOW_UP}"


class _CrisisOutput(BaseModel):
    en_peligro: bool
    motivo: str


async def crisis_check_node(state: RAGState) -> dict:
    """Detecta peligro actual o inminente. Corre en paralelo con el filtro de dominio, que no cambia: un mensaje de
    crisis nunca recibe la respuesta de fuera de dominio, y las decisiones del filtro sobre el resto quedan iguales."""
    if not settings.crisis_check_enabled:
        return {"in_crisis": False}
    llm = get_llm_structured(_CrisisOutput)
    result: _CrisisOutput = await llm.ainvoke([
        {"role": "system", "content": prompts.crisis_check},
        {"role": "user", "content": state["user_query"]},
    ])
    return {"in_crisis": result.en_peligro}


def triage_node(state: RAGState) -> dict:
    """Punto de encuentro del detector de crisis y del filtro de dominio (corren en paralelo)."""
    return {}


def crisis_response_node(state: RAGState) -> dict:
    """Mensaje de crisis que el filtro de dominio rechazó: indicaciones de seguridad en lugar de "fuera de dominio"."""
    return {
        "answer": CRISIS_ANSWER,
        "sources": [],
        "crisis": True,
        "out_of_domain": False,
        "no_results": False,
        "messages": [{"role": "assistant", "content": CRISIS_ANSWER}],
    }


def crisis_notice_node(state: RAGState) -> dict:
    """Mensaje de crisis que siguió el camino normal: la indicación de seguridad va antes de la respuesta."""
    if not state.get("in_crisis"):
        return {}
    return {"answer": f"{SAFETY_NOTICE}\n\n{state.get('answer', '')}", "crisis": True}
