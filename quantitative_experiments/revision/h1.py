"""Portable H1 evaluation: default dry-run; add --run to download models and evaluate.

python -m quantitative_experiments.revision.h1 --dry-run
python -m quantitative_experiments.revision.h1 --run --device cuda
python -m quantitative_experiments.revision.h1 --run --device mps

Auto device order: CUDA, Apple MPS, CPU. Both models use the same precision,
pooling, token limit and batches. Each execution requires a fresh output directory.
"""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    CORPUS_PATH, OUT, ROOT, bootstrap_for, load_dataset, load_qrels, paired, per_query, read_jsonl,
)

MODELS = {
    "base": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    "finetuned": ("ampara-ai/bge-m3-ampara", "f2fab1a2bf888b4da42c1660beb578eb32ead313"),
}
WEIGHTS = {
    "base": ("pytorch_model.bin", "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"),
    "finetuned": ("model.safetensors", "7902381e91104e1b6c50c1aa0a751b0b51426ebe8f5f0724dd9ebe47867bfaa1"),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Only publish a complete JSON file, even if writing is interrupted."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def detect_hardware():
    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda": False, "mps": False}
    return {"torch_available": True, "cuda": bool(torch.cuda.is_available()),
            "mps": bool(torch.backends.mps.is_available())}


def select_device(requested, *, cuda, mps):
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device: {requested}")
    if (requested == "cuda" and not cuda) or (requested == "mps" and not mps):
        raise ValueError(f"Requested device {requested} is unavailable; no automatic fallback")
    return requested


def select_precision(requested, device):
    if requested == "auto":
        return "fp32" if device == "cpu" else "fp16"
    if requested not in ("fp16", "fp32"):
        raise ValueError(f"Unknown precision: {requested}")
    if requested == "fp16" and device == "cpu":
        raise ValueError("CPU execution requires fp32")
    return requested


def validate_inputs(dataset, qrels, corpus_rows):
    qids = [d.get("dataset_id") for d in dataset]
    ids = [c.get("chunk_id") for c in corpus_rows]
    if not qids or any(not isinstance(q, str) or not q for q in qids) or len(qids) != len(set(qids)):
        raise ValueError("Dataset needs unique, nonempty query IDs")
    if not ids or any(not isinstance(c, str) or not c for c in ids) or len(ids) != len(set(ids)):
        raise ValueError("Corpus needs unique, nonempty chunk IDs")
    if any(not isinstance(c.get("text"), str) or not c["text"].strip() for c in corpus_rows):
        raise ValueError("Corpus contains empty text")
    if set(qrels) != set(qids) or any(not qrels[q] for q in qids):
        raise ValueError("Strict qrels must cover exactly every query with at least one positive")
    corpus_ids = set(ids)
    for d in dataset:
        q = d["dataset_id"]
        if not isinstance(d.get("query"), str) or not d["query"].strip():
            raise ValueError(f"Empty query: {q}")
        if not qrels[q] <= corpus_ids or d.get("source_chunk") not in qrels[q]:
            raise ValueError(f"Invalid source/qrels for {q}")
        if not isinstance(d.get("positive_chunk_ids"), list) or set(d["positive_chunk_ids"]) != qrels[q]:
            raise ValueError(f"Dataset positives disagree with strict qrels for {q}")


def validate_embeddings(vectors, count, expected_dim=1024):
    if vectors.shape != (count, expected_dim) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding shape or nonfinite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=0.02, rtol=0):
        raise ValueError("Embeddings must have unit L2 norm")


def score_pair(dataset, qrels, runs, *, resamples=10_000, seed=0):
    qids = [d["dataset_id"] for d in dataset]
    if not qids or len(set(qids)) != len(qids) or set(qrels) != set(qids):
        raise ValueError("Invalid query/qrel coverage")
    if resamples < 1 or seed < 0:
        raise ValueError("Bootstrap resamples must be positive and seed nonnegative")
    if set(runs) != {"base", "finetuned"}:
        raise ValueError("Expected exactly base and finetuned runs")
    for run in runs.values():
        if set(run) != set(qids):
            raise ValueError("Run query coverage does not match dataset")
        for ranking in run.values():
            if not isinstance(ranking, list) or any(not isinstance(c, str) or not c for c in ranking):
                raise ValueError("Ranking must contain chunk IDs")
            if len(ranking) != len(set(ranking)):
                raise ValueError("Duplicate documents in ranking")
    values = {n: per_query(r, qrels, qids, "ndcg", 10) for n, r in runs.items()}
    boot = bootstrap_for(dataset, n_resamples=resamples, seed=seed)
    difference = paired(boot, values["finetuned"], values["base"])
    replicates = boot.resampled_means(values["finetuned"] - values["base"])
    raw_p = float(min(1.0, 2 * min(np.mean(replicates <= 0), np.mean(replicates >= 0))))
    report = {
        "hypothesis": "H1", "metric": "strict NDCG@10", "queries": len(qids),
        "source_clusters": boot.n_clusters, "base_ndcg10": float(values["base"].mean()),
        "finetuned_ndcg10": float(values["finetuned"].mean()), "difference": difference,
        "raw_bootstrap_tail_p": raw_p, "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "confidence": 0.95, "multiplicity": "H1-H3 Holm adjustment pending",
        "scope": "Single fixed checkpoint; source/alias recovery on synthetic questions; no legal validation",
    }
    rows = [{"qid": q, "source_chunk": dataset[i]["source_chunk"], "n_relevant": len(qrels[q]),
             "base_ndcg10": float(values["base"][i]), "finetuned_ndcg10": float(values["finetuned"][i]),
             "difference": float(values["finetuned"][i] - values["base"][i])} for i, q in enumerate(qids)]
    return report, rows


def model_spec(name, identifier, revision):
    local = Path(identifier).expanduser()
    if local.is_dir():
        if revision:
            raise ValueError("A local checkpoint does not take a Hub revision")
        return {"local_path": str(local.resolve())}
    if local.is_absolute() or identifier.startswith("."):
        raise ValueError(f"Checkpoint folder does not exist: {identifier}")
    revision = revision or (MODELS[name][1] if identifier == MODELS[name][0] else None)
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub models require an immutable 40-character commit revision")
    return {"repository": identifier, "revision": revision}


def resolve_model(spec):
    if "local_path" in spec:
        return Path(spec["local_path"])
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        spec["repository"], revision=spec["revision"],
        allow_patterns=["config.json", "model*.safetensors", "model.safetensors.index.json",
                        "pytorch_model*.bin", "pytorch_model.bin.index.json", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"],
    ))


def make_encoder(path, device, precision, max_length):
    from quantitative_experiments.revision.embed import DenseEncoder
    return DenseEncoder(f"cls:{path}", device=device, max_length=max_length, fp16=precision == "fp16")


def release_memory(device):
    gc.collect()
    import torch
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def execute(plan, dataset, qrels, corpus_rows):
    """Called only with --run. No model or embedding is reused from previous runs."""
    output = Path(plan["output_dir"])
    if output.exists():
        raise ValueError("Output directory already exists (possibly nonempty); choose a new --out-dir (no implicit resume)")
    output.mkdir(parents=True, exist_ok=False)  # Atomic claim: concurrent launches cannot share an execution.
    stage = "prepare_models"
    write_json(output / "status.json", {"state": "running", "stage": stage, "started_at_utc": utc_now()})
    try:
        paths = {n: resolve_model(spec) for n, spec in plan["models"].items()}
        model_files = {}
        for name, path in paths.items():
            files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts)
            model_files[name] = {str(p.relative_to(path)): file_hash(p) for p in files}
            if plan["models"][name] == {"repository": MODELS[name][0], "revision": MODELS[name][1]}:
                filename, expected = WEIGHTS[name]
                if model_files[name].get(filename) != expected:
                    raise ValueError(f"Pinned {name} weights failed SHA-256 verification")
        from transformers import AutoTokenizer
        from quantitative_experiments.revision.local_retrieval import topk
        corpus_rows = sorted(corpus_rows, key=lambda c: c["chunk_id"])
        ids, texts = [c["chunk_id"] for c in corpus_rows], [c["text"] for c in corpus_rows]
        queries = [d["query"] for d in dataset]
        settings = plan["settings"]
        tokens = [AutoTokenizer.from_pretrained(str(p), local_files_only=True)(
            queries + texts, truncation=True, max_length=settings["max_length"])["input_ids"] for p in paths.values()]
        if tokens[0] != tokens[1]:
            raise ValueError("Base and fine-tuned tokenizers differ on the evaluation texts")
        del tokens
        code_paths = [Path(__file__), Path(__file__).with_name("embed.py"), Path(__file__).with_name("local_retrieval.py"),
                      Path(__file__).with_name("data.py"), ROOT / "evaluation/metrics.py", ROOT / "evaluation/bootstrap.py"]
        protocol = {**plan, "frozen_at_utc": utc_now(), "registration": "local run specification, not external preregistration",
                    "model_file_sha256": model_files, "tokenizers_identical_on_evaluation_inputs": True,
                    "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths},
                    "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                                "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
                                "packages": {n: importlib.metadata.version(n) for n in
                                             ["torch", "numpy", "sentence-transformers", "transformers", "huggingface-hub"]}}}
        write_json(output / "protocol.json", protocol)
        protocol_hash = file_hash(output / "protocol.json")
        runs = {}
        for name, path in paths.items():
            stage = name
            write_json(output / "status.json", {"state": "running", "stage": stage, "updated_at_utc": utc_now()})
            print(f"Running {name} on {settings['device']} ({settings['precision']})", flush=True)
            encoder = make_encoder(path, settings["device"], settings["precision"], settings["max_length"])
            try:
                info = encoder.info().__dict__
                if info["device"] != settings["device"] or info["dtype"] != settings["precision"]:
                    raise ValueError("Encoder device/precision differs from frozen protocol")
                started = time.perf_counter()
                docs = encoder.encode_documents(texts, settings["doc_batch"])
                query_vectors = encoder.encode_queries(queries, settings["query_batch"])
                validate_embeddings(docs, len(texts))
                validate_embeddings(query_vectors, len(queries))
                scores = encoder.scores(query_vectors, docs)
                if scores.shape != (len(queries), len(texts)) or not np.isfinite(scores).all():
                    raise ValueError("Invalid retrieval scores")
                rankings = topk(scores, ids, settings["depth"])
                expected = min(settings["depth"], len(ids))
                if any(len(r) != expected or len(r) != len(set(r)) or not set(r) <= set(ids) for r in rankings):
                    raise ValueError("Incomplete or invalid rankings")
                run = {d["dataset_id"]: r for d, r in zip(dataset, rankings)}
                write_json(output / f"{name}.json", run)
                write_json(output / f"{name}.meta.json", {**info, "seconds": time.perf_counter() - started,
                           "protocol_sha256": protocol_hash, "rankings_sha256": file_hash(output / f"{name}.json")})
                runs[name] = run
                del docs, query_vectors, scores
            finally:
                del encoder
                release_memory(settings["device"])
        stage = "comparison"
        for spec in plan["inputs"].values():
            if file_hash(spec["path"]) != spec["sha256"]:
                raise ValueError("An input changed during evaluation")
        for path, expected in protocol["code_sha256"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("Analysis code changed during evaluation")
        report, rows = score_pair(dataset, qrels, runs, resamples=settings["resamples"], seed=settings["seed"])
        report["protocol_sha256"] = protocol_hash
        report["dataset_sha256"] = plan["inputs"]["dataset"]["sha256"]
        report["qrels_sha256"] = plan["inputs"]["qrels"]["sha256"]
        write_json(output / "comparison.json", report)
        with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        delta = report["difference"]
        (output / "comparison.md").write_text(
            f"# H1: strict NDCG@10 on Test B\n\n"
            f"Base: {report['base_ndcg10']:.6f}. Fine-tuned: {report['finetuned_ndcg10']:.6f}.\n\n"
            f"FT − base: {delta['estimate']:+.6f}; 95% CI [{delta['ci_low']:.6f}, {delta['ci_high']:.6f}].\n\n"
            f"{report['queries']} queries, {report['source_clusters']} source clusters; "
            f"{settings['resamples']} paired cluster bootstrap resamples, seed {settings['seed']}.\n\n"
            "H1–H3 Holm adjustment remains pending. The CI does not include training-seed variation. "
            "This measures source recovery under a different synthetic question protocol, not legal correctness.\n",
            encoding="utf-8")
        write_json(output / "status.json", {"state": "complete", "completed_at_utc": utc_now(),
                   "protocol_sha256": protocol_hash, "artifacts": {p.name: file_hash(p) for p in output.iterdir()
                                                                  if p.is_file() and p.name != "status.json"}})
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        write_json(output / "status.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "stage": stage, "error_type": type(exc).__name__, "updated_at_utc": utc_now()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly download models and start inference")
    mode.add_argument("--dry-run", action="store_true", help="Default: validate inputs and print plan only")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--dataset", type=Path, default=OUT / "test_b/eval_test_b_dataset.jsonl")
    parser.add_argument("--qrels", type=Path, default=OUT / "test_b/eval_test_b_qrels.jsonl")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT / "h1")
    for name in MODELS:
        parser.add_argument(f"--{name}-model", default=MODELS[name][0])
        parser.add_argument(f"--{name}-revision")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--doc-batch", type=int, default=4)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.max_length, args.doc_batch, args.query_batch, args.resamples) < 1 or args.depth < 10 or args.seed < 0:
        raise ValueError("Lengths/batches/resamples must be positive, depth at least 10, seed nonnegative")
    dataset, qrels, corpus_rows = load_dataset(args.dataset), load_qrels(args.qrels), read_jsonl(args.corpus)
    validate_inputs(dataset, qrels, corpus_rows)
    hardware = detect_hardware()
    if not hardware["torch_available"] and args.run:
        raise RuntimeError("Install PyTorch and requirements-h1.txt before --run")
    if not args.run and args.device != "auto":
        device = args.device  # Preview a remote CUDA/MPS target without requiring it on this host.
    else:
        device = select_device(args.device, cuda=hardware["cuda"], mps=hardware["mps"]) if hardware["torch_available"] else None
    precision = select_precision(args.precision, device) if device else None
    plan = {
        "mode": "run" if args.run else "dry-run", "hardware": hardware,
        "models": {name: model_spec(name, getattr(args, f"{name}_model"), getattr(args, f"{name}_revision")) for name in MODELS},
        "settings": {"device": device, "device_requested": args.device, "precision": precision,
                     "device_available_locally": bool(hardware["torch_available"] and
                                                       (device == "cpu" or hardware.get(device, False))),
                     "precision_requested": args.precision, "max_length": args.max_length, "depth": args.depth,
                     "doc_batch": args.doc_batch, "query_batch": args.query_batch, "resamples": args.resamples, "seed": args.seed},
        "inputs": {name: {"path": str(path.resolve()), "sha256": file_hash(path)}
                   for name, path in {"dataset": args.dataset, "qrels": args.qrels, "corpus": args.corpus}.items()},
        "queries": len(dataset), "source_clusters": len({d["source_chunk"] for d in dataset}),
        "corpus_chunks": len(corpus_rows), "output_dir": str(args.out_dir.resolve()),
        "hypothesis": "FT minus base strict NDCG@10 on Test B is positive",
        "checkpoint_identity": "Default public release is identified as checkpoint-256 by its model card",
        "multiplicity": "H1-H3 Holm adjustment pending",
    }
    if args.run:
        execute(plan, dataset, qrels, corpus_rows)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
