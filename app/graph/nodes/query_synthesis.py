from app.core.dependencies import get_llm
from app.core.prompt_provider import prompts
from app.graph.state import RAGState


async def query_synthesis_node(state: RAGState) -> dict:
    recent = state.get("messages", [])[-3:]
    context = "\n".join(f"{m['role']}: {m['content']}" for m in recent[:-1])

    user_content = state["user_query"]
    if context:
        user_content = f"Contexto:\n{context}\n\nPregunta: {state['user_query']}"

    llm = get_llm()
    result = await llm.ainvoke(
        [
            {"role": "system", "content": prompts.query_synthesis},
            {"role": "user", "content": user_content},
        ]
    )
    synthesized = result.content.strip()
    return {"synthesized_query": synthesized}
