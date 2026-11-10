"""Reescritura de consultas con un LLM (MAJOR #2, grupo 3) y prueba del mecanismo "registro coloquial -> registro legal".

Modos:
  legal  Reescribe la pregunta coloquial como una consulta en lenguaje jurídico peruano, sin agregar hechos.
  hyde   Escribe un pasaje legal hipotético que respondería la pregunta (HyDE); se recupera con ese pasaje.

Salida: una copia del conjunto de preguntas con el campo `query_rewritten`, lista para
    local_retrieval --dataset <salida> --query-field query_rewritten
Si el fine-tuning aprendió sobre todo a cruzar registros, el modelo base debería acercarse al fine-tuned con `legal`.

Uso:
    uv run python -m quantitative_experiments.revision.query_rewrite --mode legal [--model openai/gpt-oss-120b] [--dataset ...] [--dry-run]
"""

import argparse
import asyncio
from pathlib import Path

from quantitative_experiments.revision.data import OUT, TEST_DATASET, load_dataset, write_jsonl
from quantitative_experiments.revision.llm import LLM

PROMPTS = {
    "legal": (
        "Eres abogado especialista en violencia contra las mujeres y los integrantes del grupo familiar en el Perú. "
        "Recibirás la pregunta de una persona sin formación jurídica. Reescríbela como una consulta breve en lenguaje "
        "jurídico peruano, con los términos técnicos que usarían la ley y los códigos (por ejemplo, medidas de protección, "
        "agresiones en contra de las mujeres o integrantes del grupo familiar, reparación civil, prisión preventiva). "
        "No agregues hechos ni detalles que la pregunta no tenga y no cites números de artículos. "
        'Responde SOLO con JSON: {"query": "..."}'
    ),
    "hyde": (
        "Eres abogado especialista en violencia contra las mujeres y los integrantes del grupo familiar en el Perú. "
        "Recibirás la pregunta de una persona sin formación jurídica. Escribe, en 3 a 5 oraciones, el fragmento de una norma "
        "o de una sentencia peruana que respondería esa pregunta, con el estilo y el vocabulario de un texto legal. "
        "No cites números de artículos. "
        'Responde SOLO con JSON: {"query": "..."}'
    ),
}


async def run(args) -> None:
    dataset = load_dataset(args.dataset)
    llm = LLM(args.model, OUT / "rewrite" / f"cache_{args.mode}.jsonl", temperature=0, concurrency=args.concurrency,
              dry_run=args.dry_run, max_tokens=2000)

    def check(resp):
        if not isinstance(resp.get("query"), str) or not resp["query"].strip():
            raise ValueError("sin campo query")

    responses = await asyncio.gather(*(llm.json(PROMPTS[args.mode], d["query"], validate=check) for d in dataset))
    print(llm.report())
    if args.dry_run:
        return
    out = args.out or OUT / "rewrite" / f"{args.dataset.stem}__{args.mode}__{args.model.replace('/', '_')}.jsonl"
    rows = []
    for d, r in zip(dataset, responses):
        row = {k: v for k, v in d.items() if k != "source_chunk"}
        row.update({"query_rewritten": r["query"].strip(), "rewrite": {"mode": args.mode, "model": args.model}})
        rows.append(row)
    write_jsonl(out, rows)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(PROMPTS), default="legal")
    ap.add_argument("--model", default="openai/gpt-oss-120b", help="por defecto el LLM del pipeline desplegado")
    ap.add_argument("--dataset", type=Path, default=TEST_DATASET)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
