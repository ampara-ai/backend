"""Reranker autoalojable (MAJOR #7): calidad y latencia de bge-reranker-v2-m3 sobre los K primeros candidatos de una etapa.

Para cada K en --depths reordena los K primeros de la corrida --first-stage y conserva el resto en su orden, de modo que el
ranking sigue teniendo profundidad 100 (Recall@100 igual al de la primera etapa). Escribe una corrida por K
(<name>_k<K>.json) y la latencia por consulta (mediana y percentil 95) en el dispositivo usado; con --cpu-latency-sample N
mide además N consultas en CPU, la condición del despliegue sin GPU.

Modelos:
  st:BAAI/bge-reranker-v2-m3   PyTorch (CUDA, Apple MPS o CPU; FP16 en GPU)
  onnx:<carpeta>               el export ONNX del model service (reranker_onnx_int8), siempre en CPU

Uso:
    uv run python -m quantitative_experiments.revision.rerank_local --first-stage released:dense_finetuned \
        --depths 20 50 100 --name bge_reranker_on_ft [--cpu-latency-sample 30]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    OUT, STRICT_QRELS, TEST_DATASET, corpus, load_dataset, load_qrels, load_run, released_run, summary,
)
from quantitative_experiments.revision.embed import pick_device


class Reranker:
    def __init__(self, spec: str, device: str, max_length: int, fp16: bool) -> None:
        self.type, self.name = spec.split(":", 1)
        self.max_length = max_length
        if self.type == "onnx":
            import onnxruntime as ort
            from transformers import AutoTokenizer
            folder = Path(self.name)
            self.session = ort.InferenceSession(str(next(folder.glob("*.onnx"))), providers=["CPUExecutionProvider"])
            self.tokenizer = AutoTokenizer.from_pretrained(str(folder))
            self.inputs = {i.name for i in self.session.get_inputs()}
            self.device, self.dtype = "cpu", "onnx"
        elif self.type == "st":
            import torch
            from sentence_transformers import CrossEncoder
            self.device = pick_device(device)
            half = fp16 and self.device != "cpu"
            self.model = CrossEncoder(self.name, device=self.device, max_length=max_length,
                                      model_kwargs={"torch_dtype": torch.float16} if half else {})
            self.dtype = "fp16" if half else "fp32"
        else:
            raise ValueError("usa st:<hf id> u onnx:<carpeta>")

    def score(self, query: str, docs: list[str], batch_size: int = 16) -> np.ndarray:
        if self.type == "onnx":  # mismo cálculo que el model service: sigmoide del logit
            logits = []
            for i in range(0, len(docs), batch_size):
                enc = self.tokenizer([[query, d] for d in docs[i:i + batch_size]], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="np")
                logits.append(self.session.run(None, {k: v for k, v in enc.items() if k in self.inputs})[0].reshape(-1))
            return 1 / (1 + np.exp(-np.concatenate(logits)))
        return np.asarray(self.model.predict([(query, d) for d in docs], batch_size=batch_size, show_progress_bar=False))


def rerank(rr: Reranker, query: str, ranked: list[str], k: int, texts: dict[str, str]) -> tuple[list[str], float]:
    head = ranked[:k]
    t0 = time.perf_counter()
    s = rr.score(query, [texts[c] for c in head])
    secs = time.perf_counter() - t0
    order = sorted(range(len(head)), key=lambda i: (-s[i], head[i]))
    return [head[i] for i in order] + ranked[k:], secs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="st:BAAI/bge-reranker-v2-m3")
    ap.add_argument("--first-stage", required=True, help="ruta a una corrida JSON o released:<config>")
    ap.add_argument("--name", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--dataset", type=Path, default=TEST_DATASET)
    ap.add_argument("--qrels", type=Path, default=STRICT_QRELS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-length", type=int, default=1024, help="1024, como el model service")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--cpu-latency-sample", type=int, default=0)
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    first = released_run(args.first_stage.split(":", 1)[1]) if args.first_stage.startswith("released:") else load_run(Path(args.first_stage))
    texts = {c: d["text"] for c, d in corpus().items()}
    out_dir = args.out_dir or OUT / "runs" / args.dataset.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    qrels, qids = load_qrels(args.qrels), [d["dataset_id"] for d in dataset]
    rr = Reranker(args.model, args.device, args.max_length, fp16=not args.fp32)
    meta = {"model": args.model, "device": rr.device, "dtype": rr.dtype, "first_stage": args.first_stage, "per_k": {}}

    for k in args.depths:
        run, secs = {}, []
        for d in dataset:
            run[d["dataset_id"]], s = rerank(rr, d["query"], first[d["dataset_id"]], k, texts)
            secs.append(s)
        (out_dir / f"{args.name}_k{k}.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        entry = {"latency_ms_median": 1000 * statistics.median(secs), "latency_ms_p95": 1000 * float(np.percentile(secs, 95)),
                 **summary(run, qrels, qids, ks=(10,))}
        if args.cpu_latency_sample and rr.device != "cpu":
            cpu = Reranker(args.model, "cpu", args.max_length, fp16=False)
            cs = [rerank(cpu, d["query"], first[d["dataset_id"]], k, texts)[1] for d in dataset[: args.cpu_latency_sample]]
            entry["cpu_latency_ms_median"] = 1000 * statistics.median(cs)
        meta["per_k"][k] = entry
        print(f"K={k}: " + "  ".join(f"{m}={v:.3f}" for m, v in entry.items()))
    (out_dir / f"{args.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
