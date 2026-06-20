"""
Evalúa las 4 métricas principales de Ragas (context_precision, context_recall,
faithfulness, answer_relevancy) sobre un dataset de validación tipo
{"id", "query", "pos": [...], "neg": [...]}.

Este script incluye un pipeline RAG SIMULADO construido con LangGraph
(retriever de juguete + nodo de generación que sí llama a un LLM real vía
OpenRouter) solo para que el script corra de punta a punta como referencia.
Reemplaza build_mock_rag_graph() por tu grafo real de LangGraph.

Requisitos:
    pip install ragas rapidfuzz pandas openai langgraph

----------------------------------------------------------------------------
NOTA SOBRE LA API DE RAGAS (importante, no es un descuido):
Al momento de escribir este script, Ragas migró Faithfulness y
AnswerRelevancy a la API nueva basada en "collections" (método .ascore(),
devuelve un objeto con .value), pero las versiones Non-LLM de context
precision y context recall (NonLLMContextPrecisionWithReference,
NonLLMContextRecall) TODAVÍA viven en la API legacy (SingleTurnSample +
.single_turn_ascore(), devuelve un float directo). Por eso este script usa
ambas convenciones, una para cada par de métricas. Si en el futuro
actualizas ragas y esto cambió, revisa la documentación oficial.
----------------------------------------------------------------------------
"""

import json
import re
import random
import asyncio
from typing import TypedDict

import pandas as pd
from openai import OpenAI, AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------
# 1. CARGA DEL DATASET
# ---------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------
# 2. RAG MOCK (SOLO PARA QUE EL SCRIPT CORRA; REEMPLAZA ESTO)
# ---------------------------------------------------------------------
# Construye un "corpus" de juguete a partir de los pos/neg del propio
# dataset, y un retriever de juguete basado en solapamiento de palabras.
# El parámetro 'noise' simula que un embedding peor (base) recupera con
# más error que uno mejor (fine-tuneado). Esto NO es un embedding real,
# es solo para que las 4 métricas tengan algo que medir end-to-end.

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def build_mock_corpus(dataset: list[dict]) -> list[str]:
    seen = set()
    corpus = []
    for item in dataset:
        for chunk in item.get("pos", []) + item.get("neg", []):
            if chunk not in seen:
                seen.add(chunk)
                corpus.append(chunk)
    return corpus


def make_mock_retriever(corpus: list[str], noise: float = 0.0):
    """
    Retriever de juguete (Jaccard sobre palabras). REEMPLAZA esto por tu
    retriever real (FAISS/Chroma/Qdrant + bge-m3 base o fine-tuneado).
    """
    def retriever_fn(query: str, k: int = 5) -> list[str]:
        query_tokens = _tokenize(query)
        scored = []
        for doc in corpus:
            doc_tokens = _tokenize(doc)
            if not doc_tokens or not query_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            score = overlap + random.uniform(0, noise)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]
    return retriever_fn


def make_generate_fn(client: OpenAI, model: str):
    """
    Nodo de generación REAL: llama a un LLM (vía OpenRouter) usando el
    contexto recuperado. Reemplaza 'model' por el que realmente usas.
    """
    def generate_fn(question: str, documents: list[str]) -> str:
        context = "\n\n".join(documents)
        prompt = (
            "Responde la siguiente pregunta legal usando ÚNICAMENTE la "
            "información del contexto. Si el contexto no es suficiente, "
            "dilo explícitamente.\n\n"
            f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    return generate_fn


class RAGState(TypedDict):
    question: str
    documents: list[str]
    answer: str


def build_mock_rag_graph(retriever_fn, generate_fn, k: int = 5):
    def retrieve_node(state: RAGState) -> dict:
        docs = retriever_fn(state["question"], k=k)
        return {"documents": docs}

    def generate_node(state: RAGState) -> dict:
        answer = generate_fn(state["question"], state["documents"])
        return {"answer": answer}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def make_rag_fn(graph):
    """Adapta tu grafo (real o mock) a (retrieved_contexts, response)."""
    def rag_fn(query: str):
        result = graph.invoke({"question": query})
        return result["documents"], result["answer"]
    return rag_fn


# ---------------------------------------------------------------------
# 3. EVALUACIÓN CON LAS 4 MÉTRICAS DE RAGAS
# ---------------------------------------------------------------------

METRIC_COLUMNS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


async def evaluate_dataset(
    dataset: list[dict],
    rag_fn,
    evaluator_llm,
    evaluator_embeddings,
    label: str = "modelo",
) -> tuple[dict, pd.DataFrame]:
    # --- Non-LLM, API legacy: comparan retrieved_contexts vs reference_contexts ---
    context_precision_metric = NonLLMContextPrecisionWithReference()
    context_recall_metric = NonLLMContextRecall()

    # --- LLM-based, API moderna (collections) ---
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

    rows = []
    for item in dataset:
        query = item["query"]
        reference_contexts = item["pos"]  # ground truth para precision/recall

        retrieved_contexts, response = rag_fn(query)

        # context_precision / context_recall (legacy API)
        retrieval_sample = SingleTurnSample(
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )
        precision_value = await context_precision_metric.single_turn_ascore(retrieval_sample)
        recall_value = await context_recall_metric.single_turn_ascore(retrieval_sample)

        # faithfulness / answer_relevancy (API moderna)
        faithfulness_result = await faithfulness_metric.ascore(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        relevancy_result = await answer_relevancy_metric.ascore(
            user_input=query,
            response=response,
        )

        rows.append({
            "id": item.get("id"),
            "query": query,
            "context_precision": precision_value,
            "context_recall": recall_value,
            "faithfulness": faithfulness_result.value,
            "answer_relevancy": relevancy_result.value,
            "topic": item.get("topic"),
            "barrier": item.get("barrier"),
        })

    df = pd.DataFrame(rows)
    summary = {"modelo": label, "n_queries": len(df)}
    for col in METRIC_COLUMNS:
        summary[f"mean_{col}"] = df[col].mean()
        summary[f"std_{col}"] = df[col].std()

    return summary, df


# ---------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------

async def main():
    dataset_path = "validacion.jsonl"  # tu ruta real
    dataset = load_jsonl(dataset_path)

    OPENROUTER_API_KEY = "<OPENROUTER_API_KEY>"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Cliente ASYNC: usado por Ragas (evaluator_llm + evaluator_embeddings)
    async_client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={  # opcional, solo afecta el ranking en openrouter.ai
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>",
        },
    )

    # Cliente SYNC: usado dentro del nodo de generación del grafo (LangGraph
    # invoca de forma síncrona en este ejemplo)
    sync_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    # --- LLM "juez" para faithfulness/answer_relevancy ---
    # provider="openai" le dice a Ragas que trate el endpoint como
    # compatible con la API de OpenAI (que es justo lo que es OpenRouter).
    # Usa un modelo con buen soporte de tool-calling/structured output
    # (Ragas necesita eso internamente); si tu modelo elegido falla con
    # errores de parseo, prueba con otro disponible en openrouter.ai/models.
    evaluator_llm = llm_factory(
        "qwen/qwen3.7-plus",
        provider="openai",
        client=async_client,
    )

    # --- Embeddings "juez" para answer_relevancy ---
    # Verifica el slug exacto del modelo en openrouter.ai/models antes de correr.
    evaluator_embeddings = OpenAIEmbeddings(
        client=async_client,
        model="google/gemini-embedding-2",
    )

    # --- Modelo de generación real de tu RAG (puede ser distinto al juez) ---
    generation_model = "google/gemini-2.0-flash"  # cambia por el que uses en tu RAG

    # --- Construcción del RAG mock (reemplaza por tus grafos reales) ---
    corpus = build_mock_corpus(dataset)
    retriever_base_mock = make_mock_retriever(corpus, noise=0.3)       # simula embedding base (peor)
    retriever_finetuned_mock = make_mock_retriever(corpus, noise=0.05)  # simula embedding fine-tuneado (mejor)
    generate_fn = make_generate_fn(sync_client, generation_model)

    graph_base = build_mock_rag_graph(retriever_base_mock, generate_fn)
    graph_finetuned = build_mock_rag_graph(retriever_finetuned_mock, generate_fn)

    rag_fn_base = make_rag_fn(graph_base)
    rag_fn_finetuned = make_rag_fn(graph_finetuned)

    # --- Evaluación de ambos sistemas con las mismas 4 métricas ---
    summary_base, df_base = await evaluate_dataset(
        dataset, rag_fn_base, evaluator_llm, evaluator_embeddings, label="embedding_base"
    )
    summary_ft, df_ft = await evaluate_dataset(
        dataset, rag_fn_finetuned, evaluator_llm, evaluator_embeddings, label="embedding_finetuned"
    )

    comparison = pd.DataFrame([summary_base, summary_ft])
    print(comparison.to_string(index=False))

    df_base.to_csv("resultados_base_4_metricas.csv", index=False)
    df_ft.to_csv("resultados_finetuned_4_metricas.csv", index=False)
    comparison.to_csv("resumen_comparacion_4_metricas.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())