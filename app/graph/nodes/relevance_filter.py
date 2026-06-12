from pydantic import BaseModel

from app.core.config import settings
from app.core.dependencies import get_llm_structured
from app.core.prompt_provider import prompts
from app.graph.state import RAGState


class _RelevanceOutput(BaseModel):
    is_relevant: bool
    reason: str


async def relevance_filter_node(state: RAGState) -> dict:
    recent = state.get("messages", [])[-settings.session_max_messages :]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent[:-1])

    user_content = state["user_query"]
    if history_text:
        user_content = f"Contexto previo:\n{history_text}\n\nPregunta actual: {state['user_query']}"

    llm = get_llm_structured(_RelevanceOutput)
    messages = [
        {"role": "system", "content": prompts.relevance_filter},
        {"role": "user", "content": user_content},
    ]
    result: _RelevanceOutput = await llm.ainvoke(messages)
    return {"is_relevant": result.is_relevant}
