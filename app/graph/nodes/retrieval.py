from app.graph.state import RAGState
from app.services import mongodb


async def bm25_search_node(state: RAGState) -> dict:
    results = await mongodb.bm25_search(state["synthesized_query"], k=20)
    return {"bm25_results": results}


async def semantic_search_node(state: RAGState) -> dict:
    results = await mongodb.vector_search(state["query_embedding"], k=20)
    return {"semantic_results": results}
