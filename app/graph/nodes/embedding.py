from app.graph.state import RAGState
from app.services import model_service


async def embed_query_node(state: RAGState) -> dict:
    embedding = await model_service.get_embedding(state["synthesized_query"])
    return {"query_embedding": embedding}
