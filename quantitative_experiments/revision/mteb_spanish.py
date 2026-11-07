"""¿El fine-tuning dañó la recuperación general en español? (MAJOR #9): base vs fine-tuned en tareas de recuperación de MTEB
en español. Una caída pequeña es aceptable si se reporta.

Tareas por defecto (subconjunto español, todas pequeñas): SpanishPassageRetrievalS2P, SpanishPassageRetrievalS2S,
PublicHealthQA, MultilingualNanoNQRetrieval, MintakaRetrieval. --list muestra todas las tareas de recuperación en español.

Requiere el grupo de dependencias `experiments` (uv sync --group experiments). Corre en CUDA, Apple MPS o CPU.

Uso:
    uv run --group experiments python -m quantitative_experiments.revision.mteb_spanish \
        --model base=cls:BAAI/bge-m3 --model finetuned=cls:<checkpoint-256>
"""

import argparse
import json

from quantitative_experiments.revision.data import OUT
from quantitative_experiments.revision.embed import DenseEncoder

DEFAULT_TASKS = ["SpanishPassageRetrievalS2P", "SpanishPassageRetrievalS2S", "PublicHealthQA", "MultilingualNanoNQRetrieval",
                 "MintakaRetrieval"]


def main() -> None:
    import mteb
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="nombre=especificación (cls:/st:), repetible")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-length", type=int, default=512, help="las tareas de MTEB son de pasajes cortos")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for t in mteb.get_tasks(languages=["spa"], task_types=["Retrieval"]):
            print(t.metadata.name)
        return
    tasks = mteb.get_tasks(tasks=args.tasks, languages=["spa"])
    out_dir = OUT / "mteb_spanish"
    scores: dict[str, dict[str, float]] = {}
    for spec in args.model:
        name, model_spec = spec.split("=", 1)
        enc = DenseEncoder(model_spec, args.device, args.max_length)
        result = mteb.evaluate(enc.model, tasks, encode_kwargs={"batch_size": args.batch_size},
                               cache=mteb.ResultCache(cache_path=out_dir / "cache" / name))
        scores[name] = {tr.task_name: float(tr.get_score()) for tr in result.task_results}
        print(name, scores[name])
    names = list(scores)
    lines = ["| Task (main score) | " + " | ".join(names) + (" | Difference |" if len(names) == 2 else " |"),
             "| --- |" + " --- |" * (len(names) + (1 if len(names) == 2 else 0))]
    for task in sorted({t for s in scores.values() for t in s}):
        vals = [scores[n].get(task) for n in names]
        row = f"| {task} | " + " | ".join(f"{v:.4f}" if v is not None else "–" for v in vals)
        if len(names) == 2 and None not in vals:
            row += f" | {vals[1] - vals[0]:+.4f}"
        lines.append(row + " |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
