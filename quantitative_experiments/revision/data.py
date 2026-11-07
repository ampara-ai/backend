"""Rutas, carga de insumos y puntuación compartidas por los scripts de la revisión Q1.

Todo se lee de archivos versionados (rankings, qrels, conjuntos de prueba, corpus): ningún script de puntuación toca MongoDB,
el model service ni un LLM. Las métricas son las de evaluation/metrics.py con Recall@k normalizado por min(|relevantes|, k),
la definición del artículo, y los intervalos son el bootstrap pareado por chunk de origen de evaluation/bootstrap.py.
"""

import ast
import json
import os
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from evaluation.bootstrap import ClusterBootstrap
from evaluation.metrics import average_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k

ROOT = Path(__file__).resolve().parents[2]
# claves de API del .env local (no versionado); las variables ya definidas en el entorno tienen prioridad
load_dotenv(ROOT / ".env", override=False)
EXP = ROOT / "quantitative_experiments"
RESULTS = EXP / "results"
OUT = RESULTS / "revision"
CORPUS_PATH = EXP / "data" / "inputs" / "document_chunks_all.jsonl"
TEST_DATASET = EXP / "test_set" / "eval_test_dataset.jsonl"
STRICT_QRELS = EXP / "test_set" / "eval_test_qrels.jsonl"
POOLED_QRELS = EXP / "pooling" / "output" / "eval_test_qrels_pooled.jsonl"
POOLED_PARTIAL_QRELS = EXP / "pooling" / "output" / "eval_test_qrels_pooled_parcial.jsonl"

CONFIGS = [
    "bm25_only", "dense_base", "dense_finetuned", "hybrid_base", "hybrid_finetuned",
    "dense_base_rerank100", "dense_finetuned_rerank100", "hybrid_base_rerank100", "hybrid_finetuned_rerank100",
]
METRICS = {"recall": lambda r, rel, k: recall_at_k(r, rel, k, capped=True), "mrr": reciprocal_rank_at_k,
           "ndcg": ndcg_at_k, "map": average_precision_at_k}

# Mismo modelo con dos grafías en los registros de generación.
GENERATOR_ALIASES = {"anthropic/claude-sonnet-5.5": "claude-sonnet-5-5"}


def data_repo() -> Path:
    """Repositorio de datos (data_collection_and_fine-tuning): AMPARA_DATA_REPO, o un clon junto a este repositorio."""
    candidates = [os.environ.get("AMPARA_DATA_REPO"), ROOT.parent / "data_collection_and_fine-tuning", ROOT.parent / "data"]
    for c in candidates:
        if c and (Path(c) / "generacion_dataset").is_dir():
            return Path(c)
    raise FileNotFoundError("no encuentro el repositorio de datos: define AMPARA_DATA_REPO=<ruta del clon>")


def dataset_v2() -> Path:
    return data_repo() / "generacion_dataset" / "data" / "output" / "dataset_v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def as_list(value) -> list:
    """Los conjuntos de prueba guardan algunas listas como texto ("['a', 'b']")."""
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def as_dict(value) -> dict:
    return ast.literal_eval(value) if isinstance(value, str) else dict(value)


@cache
def corpus() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in read_jsonl(CORPUS_PATH)}


def load_dataset(path: Path = TEST_DATASET) -> list[dict]:
    """Preguntas con `dataset_id`, `query`, `positive_chunk_ids` (lista) y `source_chunk` (el chunk de origen = clúster)."""
    rows = read_jsonl(path)
    for r in rows:
        r["positive_chunk_ids"] = as_list(r["positive_chunk_ids"])
        r["source_chunk"] = r["positive_chunk_ids"][0]
        if "generation" in r:
            r["generation"] = as_dict(r["generation"])
    return rows


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for r in read_jsonl(path):
        if r.get("relevance", 1) > 0:
            qrels.setdefault(r["qid"], set()).add(r["docid"])
    return qrels


def load_run(path: Path) -> dict[str, list[str]]:
    """Ranking guardado: {dataset_id: [chunk_id, ...]} (formato de results/runs/<config>.json)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def released_run(config: str, folder: str = "runs") -> dict[str, list[str]]:
    return load_run(RESULTS / folder / f"{config}.json")


@cache
def chunk_split() -> dict:
    return json.loads((dataset_v2() / "chunk_split.json").read_text(encoding="utf-8"))


def partition(chunk_id: str) -> str:
    """train / dev / test, o "excluded" (los 7 chunks cortos que no entraron al split)."""
    return chunk_split()["assignments"].get(chunk_id, "excluded")


def per_query(run: dict[str, list[str]], qrels: dict[str, set[str]], qids: Sequence[str], metric: str, k: int) -> np.ndarray:
    fn = METRICS[metric]
    return np.array([fn(run.get(q, []), qrels.get(q, set()), k) for q in qids], dtype=float)


def summary(run, qrels, qids, ks=(10, 100)) -> dict[str, float]:
    return {f"{m}@{k}": float(per_query(run, qrels, qids, m, k).mean()) for k in ks for m in METRICS}


def bootstrap_for(dataset: list[dict], n_resamples: int = 10_000, seed: int = 0) -> ClusterBootstrap:
    """Remuestreo por chunk de origen (los mismos remuestreos para todas las comparaciones, como en el artículo)."""
    return ClusterBootstrap([d["source_chunk"] for d in dataset], n_resamples=n_resamples, seed=seed)


def paired(boot: ClusterBootstrap, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return boot.paired_difference(a, b)


def fmt_ci(d: dict[str, float]) -> str:
    return f"{d['estimate']:+.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def filter_run(run: dict[str, list[str]], keep) -> dict[str, list[str]]:
    return {q: [c for c in r if keep(c)] for q, r in run.items()}
