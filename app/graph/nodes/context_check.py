from pydantic import BaseModel

from app.core.dependencies import get_llm_structured
from app.core.prompt_provider import prompts
from app.graph.state import RAGState


class _ContextOutput(BaseModel):
    sufficient: bool
    reason: str


async def context_check_node(state: RAGState) -> dict:
    messages = state.get("messages", [])

    # Con menos de 2 mensajes nunca hay suficiente contexto previo
    if len(messages) < 2:
        return {"context_sufficient": False}

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1])
    user_content = (
        f"Historial:\n{history_text}\n\n"
        f"Pregunta actual: {state['user_query']}\n\n"
        "¿El historial ya contiene suficiente información para responder esta pregunta?"
    )

    llm = get_llm_structured(_ContextOutput)
    result: _ContextOutput = await llm.ainvoke(
        [
            {"role": "system", "content": prompts.context_check},
            {"role": "user", "content": user_content},
        ]
    )
    return {"context_sufficient": result.sufficient}
