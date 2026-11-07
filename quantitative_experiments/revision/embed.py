"""Codificadores para la evaluación local (búsqueda exacta, sin MongoDB ni model service).

Especificación de modelo `<tipo>:<id o ruta>`:
  cls:<hf id | carpeta>          Pooling CLS + normalización L2, como FlagEmbedding y el model service. Para BGE-M3 base
                                 (cls:BAAI/bge-m3) y para los checkpoints de fine-tuning (carpeta de FlagEmbedding).
  st:<hf id>                     SentenceTransformer con su propia configuración, más el prefijo de consulta y de documento
                                 que recomienda cada modelo (QUERY_PROMPTS); los modelos sin entrada no usan prefijo.
  onnx:<carpeta>                 Export ONNX (p. ej. INT8) con el mismo pooling CLS del model service: reproduce las corridas
                                 publicadas sin MongoDB.
  bgem3-sparse:<hf id | carpeta> Pesos léxicos de BGE-M3 (sparse_linear.pt).
  bgem3-colbert:<hf id | carpeta> Vectores multi-token de BGE-M3 (colbert_linear.pt), puntaje max-sim.

Las cabezas sparse y ColBERT de BGE-M3 se entrenaron solo en el modelo base (el fine-tuning usó unified_finetuning=False):
son líneas de base del base, no del fine-tuned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Instrucción de tarea para los modelos que la usan (en inglés, como recomiendan Qwen3-Embedding y E5-instruct).
TASK = "Given a question written by a citizen in Spanish, retrieve the Peruvian legal passage that answers it"
QUERY_PROMPTS = {
    "Qwen/Qwen3-Embedding-0.6B": {"query": f"Instruct: {TASK}\nQuery:", "document": ""},
    "Qwen/Qwen3-Embedding-4B": {"query": f"Instruct: {TASK}\nQuery:", "document": ""},
    "Qwen/Qwen3-Embedding-8B": {"query": f"Instruct: {TASK}\nQuery:", "document": ""},
    "intfloat/multilingual-e5-large-instruct": {"query": f"Instruct: {TASK}\nQuery: ", "document": ""},
    "intfloat/multilingual-e5-large": {"query": "query: ", "document": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "document": "passage: "},
}
# Argumentos extra de encode() por modelo (jina-v3 elige el adaptador LoRA por tarea).
ENCODE_KWARGS = {
    "jinaai/jina-embeddings-v3": {"query": {"task": "retrieval.query", "prompt_name": "retrieval.query"},
                                  "document": {"task": "retrieval.passage", "prompt_name": "retrieval.passage"}},
}
TRUST_REMOTE_CODE = {"jinaai/jina-embeddings-v3", "Alibaba-NLP/gte-multilingual-base"}


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EncoderInfo:
    spec: str
    kind: str
    device: str
    dtype: str
    max_length: int
    n_params: int | None = None
    prompts: dict = field(default_factory=dict)


class DenseEncoder:
    """Embeddings densos normalizados: score = producto punto (coseno)."""

    kind = "dense"

    def __init__(self, spec: str, device: str = "auto", max_length: int = 2048, fp16: bool = True) -> None:
        self.spec, (self.type, self.name) = spec, spec.split(":", 1)
        self.device = "cpu" if self.type == "onnx" else pick_device(device)
        self.max_length = max_length
        self.prompts = QUERY_PROMPTS.get(self.name, {"query": "", "document": ""})
        self.kwargs = ENCODE_KWARGS.get(self.name, {"query": {}, "document": {}})
        if self.type == "onnx":
            self._init_onnx()
            self.dtype = "onnx"
        else:
            self._init_st(fp16 and self.device != "cpu")

    def _init_st(self, fp16: bool) -> None:
        import torch
        from sentence_transformers import SentenceTransformer, models
        model_kwargs = {"torch_dtype": torch.float16} if fp16 else {}
        if self.type == "cls":
            transformer = models.Transformer(self.name, max_seq_length=self.max_length, model_args=model_kwargs)
            pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")
            self.model = SentenceTransformer(modules=[transformer, pooling, models.Normalize()], device=self.device)
        else:
            self.model = SentenceTransformer(self.name, device=self.device, model_kwargs=model_kwargs,
                                             trust_remote_code=self.name in TRUST_REMOTE_CODE)
            self.model.max_seq_length = min(self.max_length, self.model.max_seq_length or self.max_length)
        self.dtype = "fp16" if fp16 else "fp32"
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def _init_onnx(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        folder = Path(self.name)
        self.session = ort.InferenceSession(str(next(folder.glob("*.onnx"))), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder))
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.n_params = None

    def info(self) -> EncoderInfo:
        return EncoderInfo(self.spec, self.kind, self.device, self.dtype, self.max_length, self.n_params, self.prompts)

    def _encode(self, texts: list[str], role: str, batch_size: int) -> np.ndarray:
        if self.type == "onnx":  # una a una, como el model service (sin padding entre textos)
            out = []
            for t in texts:
                enc = self.tokenizer(t, padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
                cls = self.session.run(None, {k: v for k, v in enc.items() if k in self.inputs})[0][:, 0, :]
                out.append((cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-8))[0])
            return np.asarray(out, dtype=np.float32)
        prompt = self.prompts.get(role) or None
        emb = self.model.encode(texts, batch_size=batch_size, prompt=prompt, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=len(texts) > 50, **self.kwargs.get(role, {}))
        return np.asarray(emb, dtype=np.float32)

    def encode_documents(self, texts: list[str], batch_size: int = 4) -> np.ndarray:
        return self._encode(texts, "document", batch_size)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, "query", batch_size)

    def scores(self, q: np.ndarray, d: np.ndarray) -> np.ndarray:
        return q @ d.T


class BGEM3MultiEncoder:
    """Modos sparse y ColBERT de BGE-M3, con las cabezas publicadas del modelo base (como FlagEmbedding.BGEM3FlagModel)."""

    def __init__(self, spec: str, device: str = "auto", max_length: int = 2048, fp16: bool = True) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.spec, (self.type, self.name) = spec, spec.split(":", 1)
        self.kind = "sparse" if self.type == "bgem3-sparse" else "colbert"
        self.device, self.max_length = pick_device(device), max_length
        self.fp16 = fp16 and self.device != "cpu"
        self.dtype = "fp16" if self.fp16 else "fp32"
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name, torch_dtype=torch.float16 if self.fp16 else torch.float32).to(self.device).eval()
        hidden = self.model.config.hidden_size
        head_file = "sparse_linear.pt" if self.kind == "sparse" else "colbert_linear.pt"
        self.head = torch.nn.Linear(hidden, 1 if self.kind == "sparse" else hidden)
        self.head.load_state_dict(torch.load(self._file(head_file), map_location="cpu"))
        self.head = self.head.to(self.device, dtype=self.model.dtype).eval()
        t = self.tokenizer
        self.special = {i for i in (t.cls_token_id, t.eos_token_id, t.pad_token_id, t.unk_token_id) if i is not None}
        self.n_params = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.head.parameters())
        self.prompts = {}

    def _file(self, name: str) -> str:
        local = Path(self.name) / name
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.name, name)

    def info(self) -> EncoderInfo:
        return EncoderInfo(self.spec, self.kind, self.device, self.dtype, self.max_length, self.n_params, {})

    def _encode(self, texts: list[str], batch_size: int) -> list:
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                if self.kind == "sparse":
                    weights = torch.relu(self.head(hidden)).squeeze(-1).float().cpu().numpy()
                    ids = enc["input_ids"].cpu().numpy()
                    for w_row, id_row in zip(weights, ids):
                        out.append(sparse_weights(id_row, w_row, self.special))
                else:
                    vecs = self.head(hidden[:, 1:]) * enc["attention_mask"][:, 1:, None]
                    vecs = torch.nn.functional.normalize(vecs.float(), dim=-1)
                    lengths = enc["attention_mask"].sum(1).cpu().tolist()
                    out += [v[: n - 1].half().cpu() for v, n in zip(vecs, lengths)]
        return out

    def encode_documents(self, texts: list[str], batch_size: int = 4) -> list:
        return self._encode(texts, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int = 16) -> list:
        return self._encode(texts, batch_size)

    def scores(self, q: list, d: list) -> np.ndarray:
        if self.kind == "sparse":
            return np.array([[sparse_score(qi, di) for di in d] for qi in q], dtype=np.float32)
        import torch
        docs = [x.to(self.device, dtype=torch.float32) for x in d]
        return np.array([[colbert_score(qi.to(self.device, dtype=torch.float32), di) for di in docs] for qi in q],
                        dtype=np.float32)


def sparse_weights(token_ids, weights, special: set[int]) -> dict[int, float]:
    """Peso léxico por token: el máximo sobre sus apariciones, sin tokens especiales (como BGEM3FlagModel)."""
    out: dict[int, float] = {}
    for tid, w in zip(token_ids, weights):
        tid = int(tid)
        if tid in special or w <= 0:
            continue
        out[tid] = max(out.get(tid, 0.0), float(w))
    return out


def sparse_score(q: dict[int, float], d: dict[int, float]) -> float:
    return float(sum(w * d[t] for t, w in q.items() if t in d))


def colbert_score(q, d) -> float:
    """Max-sim: para cada token de la consulta, su mejor coincidencia en el documento; promedio sobre la consulta."""
    return float((q @ d.T).max(dim=1).values.mean())


def make_encoder(spec: str, device: str = "auto", max_length: int = 2048, fp16: bool = True):
    if spec.startswith(("bgem3-sparse:", "bgem3-colbert:")):
        return BGEM3MultiEncoder(spec, device, max_length, fp16)
    if spec.split(":", 1)[0] not in ("cls", "st", "onnx"):
        raise ValueError(f"tipo de modelo desconocido en {spec!r}: usa cls:, st:, onnx:, bgem3-sparse: o bgem3-colbert:")
    return DenseEncoder(spec, device, max_length, fp16)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
