"""Elige el checkpoint de un entrenamiento con el conjunto de DESARROLLO (nunca con el de prueba), como en la Tabla 7:
el de mayor NDCG@10 estricto. Corre en CUDA, Apple MPS o CPU.

Uso (una corrida por semilla):
    uv run python -m quantitative_experiments.revision.select_checkpoint --run-dir <OUTPUT_DIR de run_training.sh> \
        [--include-base] [--device auto]
Escribe <run-dir>/checkpoint_selection.md y .json, y el checkpoint elegido en selected_checkpoint.txt.
"""

import argparse
import json
import re
from pathlib import Path

from quantitative_experiments.revision.data import dataset_v2, load_dataset, load_qrels, per_query
from quantitative_experiments.revision.local_retrieval import retrieve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=None, help="por defecto dataset_v2/eval_dev_dataset.jsonl")
    ap.add_argument("--qrels", type=Path, default=None, help="por defecto dataset_v2/eval_dev_qrels.jsonl")
    ap.add_argument("--include-base", action="store_true", help="agrega BAAI/bge-m3 como referencia")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset or dataset_v2() / "eval_dev_dataset.jsonl")
    qrels = load_qrels(args.qrels or dataset_v2() / "eval_dev_qrels.jsonl")
    qids = [d["dataset_id"] for d in dataset]
    ckpts = sorted((p for p in args.run_dir.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(re.sub(r"\D", "", p.name)))
    if (args.run_dir / "config.json").exists():
        ckpts.append(args.run_dir)  # el modelo final que FlagEmbedding guarda en la raíz
    specs = (["cls:BAAI/bge-m3"] if args.include_base else []) + [f"cls:{p}" for p in ckpts]
    if not ckpts:
        raise SystemExit(f"no hay checkpoint-* en {args.run_dir}")

    rows = []
    for spec in specs:
        run, _, _ = retrieve(spec, dataset, device=args.device, fp16=not args.fp32)
        row = {"model": spec, "ndcg@10": float(per_query(run, qrels, qids, "ndcg", 10).mean()),
               "recall@10": float(per_query(run, qrels, qids, "recall", 10).mean()),
               "recall@1": float(per_query(run, qrels, qids, "recall", 1).mean())}
        rows.append(row)
        print(row)
    candidates = [r for r in rows if r["model"] != "cls:BAAI/bge-m3"]
    best = max(candidates, key=lambda r: (r["ndcg@10"], r["recall@10"]))  # empate: mayor Recall@10
    lines = [f"# Checkpoint selection on the development set ({len(qids)} questions, strict qrels)", "",
             "| Model | NDCG@10 | Recall@10 | Recall@1 |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['model']}{' (selected)' if r is best else ''} | {r['ndcg@10']:.3f} | {r['recall@10']:.3f} | {r['recall@1']:.3f} |"
              for r in rows]
    (args.run_dir / "checkpoint_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.run_dir / "checkpoint_selection.json").write_text(json.dumps({"rows": rows, "selected": best["model"]}, indent=1),
                                                            encoding="utf-8")
    (args.run_dir / "selected_checkpoint.txt").write_text(best["model"].split(":", 1)[1] + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
