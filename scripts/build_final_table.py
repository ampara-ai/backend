"""
Tabla final consolidada para el protocolo de tesis (Objetivo Específico 4).

Une en una sola tabla (CSV + Excel) los resultados de los dos scripts de
evaluación, todos leídos/escritos en la carpeta results/ (gitignored):
  - results/comparacion_ablation.csv   (eval_ablation.py: 5 configs del pipeline)
  - results/resultados_finetuned.csv   (eval_dataset_ragas.py --model finetuned)
  - results/resultados_base.csv        (eval_dataset_ragas.py --model base)

Filas: bm25_only, dense_base, dense_finetuned, hybrid_no_rerank, hybrid_rerank,
       sistema_finetuned, sistema_base.

Columnas = las 8 métricas del protocolo de tesis:
  Dimensión "Recuperación": Recall@10, Recall@100, MRR@10, MRR@100,
    NDCG@10, NDCG@100, MAP@10, MAP@100.
  Dimensión "Calidad RAG": Faithfulness, Answer Relevancy,
    Context Recall, Context Precision.

IMPORTANTE — no todas las celdas tienen sentido en todas las filas, porque las
dos fuentes miden con metodologías distintas:
  - Las 5 filas de ablación (eval_ablation.py) traen Recall/MRR/NDCG/MAP @10 y
    @100, y Faithfulness/Answer Relevancy. Sus columnas Context Recall/Context
    Precision son EXACTAMENTE Recall@10/MAP@10 repetidas bajo el nombre
    "clásico" de RAGAS (decisión del asesor) — no una fórmula distinta ni un
    cálculo aparte. Documentar esto mismo en el texto de la tesis: se repiten
    a propósito.
  - Las 2 filas de sistema completo (sistema_finetuned/sistema_base, de
    eval_dataset_ragas.py) traen Context Recall (=recall_candidates), Context
    Precision (=context_precision_cond) — fórmulas propias de ese script,
    NO equivalentes a Recall@10/MAP@10 (candidate pool pre-RRF vs RRF top-10
    condicionado) — y Faithfulness/Answer Relevancy. Quedan vacías en
    Recall/MRR/NDCG/MAP @10/@100: ese script mide con cortes "pre-reranker"
    (RRF top-10) / "post-reranker" (top-5), no con @10/@100, y nunca calcula
    NDCG — forzar esos números en las columnas @k sería mezclar definiciones
    distintas bajo el mismo nombre.

Cada fila solo se agrega si su CSV de origen ya existe (corre primero
eval_ablation.py y/o eval_dataset_ragas.py --model finetuned/base con
--samples 100 para resultados reales).

Uso:
    uv run python scripts/build_final_table.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
ROOT.mkdir(parents=True, exist_ok=True)

ABLATION_CONFIGS = [
    "bm25_only",
    "dense_base",
    "dense_finetuned",
    "hybrid_no_rerank",
    "hybrid_rerank",
]

METRIC_COLS = [
    "Recall@10", "Recall@100", "MRR@10", "MRR@100", "NDCG@10", "NDCG@100",
    "MAP@10", "MAP@100", "Faithfulness", "Answer Relevancy",
    "Context Recall", "Context Precision",
]


def _row_from_ablation(comparacion: pd.DataFrame, config: str) -> dict:
    r = comparacion.loc[comparacion["config"] == config].iloc[0]
    return {
        "configuracion": config,
        "Recall@10": r["mean_recall_at10"],
        "Recall@100": r["mean_recall_at100"],
        "MRR@10": r["mean_mrr_at10"],
        "MRR@100": r["mean_mrr_at100"],
        "NDCG@10": r["mean_ndcg_at10"],
        "NDCG@100": r["mean_ndcg_at100"],
        "MAP@10": r["mean_map_at10"],
        "MAP@100": r["mean_map_at100"],
        "Faithfulness": r["mean_faithfulness"],
        "Answer Relevancy": r["mean_answer_relevancy"],
        # Decisión del asesor: mostrar Context Recall/Context Precision
        # también para las filas de ablación, iguales a Recall@10/MAP@10 —
        # es la misma fórmula/valor bajo el nombre "clásico" de RAGAS, no un
        # cálculo aparte (documentar esto mismo en el texto de la tesis: se
        # repiten a propósito, no es un error).
        "Context Recall": r["mean_recall_at10"],
        "Context Precision": r["mean_map_at10"],
    }


def _row_from_system(df: pd.DataFrame, label: str) -> dict:
    return {
        "configuracion": label,
        "Recall@10": None,
        "Recall@100": None,
        "MRR@10": None,
        "MRR@100": None,
        "NDCG@10": None,
        "NDCG@100": None,
        "MAP@10": None,
        "MAP@100": None,
        "Faithfulness": df["faithfulness"].mean(),
        "Answer Relevancy": df["answer_relevancy"].mean(),
        "Context Recall": df["recall_candidates"].mean(),
        "Context Precision": df["context_precision_cond"].mean(),
    }


def main() -> None:
    rows = []

    ablation_path = ROOT / "comparacion_ablation.csv"
    if ablation_path.exists():
        comparacion = pd.read_csv(ablation_path)
        for config in ABLATION_CONFIGS:
            if config in comparacion["config"].values:
                rows.append(_row_from_ablation(comparacion, config))
            else:
                print(f"  ⚠ '{config}' no está en {ablation_path.name} — fila omitida")
    else:
        print(f"  ⚠ No encontrado: {ablation_path.name} — corre scripts/eval_ablation.py primero. Filas de ablación omitidas.")

    for label, fname, model_flag in [
        ("sistema_finetuned", "resultados_finetuned.csv", "finetuned"),
        ("sistema_base", "resultados_base.csv", "base"),
    ]:
        path = ROOT / fname
        if path.exists():
            df = pd.read_csv(path)
            rows.append(_row_from_system(df, label))
        else:
            print(f"  ⚠ No encontrado: {fname} — corre scripts/eval_dataset_ragas.py --model {model_flag} primero. Fila omitida.")

    if not rows:
        print("\nNo hay ningún CSV de resultados disponible todavía. Corre las evaluaciones primero.")
        sys.exit(1)

    final = pd.DataFrame(rows, columns=["configuracion"] + METRIC_COLS)
    csv_path = ROOT / "tabla_resultados_oe4.csv"
    xlsx_path = ROOT / "tabla_resultados_oe4.xlsx"
    final.to_csv(csv_path, index=False)
    final.to_excel(xlsx_path, index=False, sheet_name="OE4")

    print(f"\nTabla final (CSV): {csv_path}")
    print(f"Tabla final (Excel): {xlsx_path}")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
