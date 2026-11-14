"""Sentence embeddings with all-MiniLM-L6-v2 (ONNX), mean-pooled and normalised."""
import os
import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models", "minilm"))
_sess = None
_tok = None


def _load():
    global _sess, _tok
    if _sess is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        _tok.enable_truncation(256)
        _tok.enable_padding()
        _sess = ort.InferenceSession(os.path.join(MODEL_DIR, os.environ.get("MODEL_FILE", "model.int8.onnx")), opts, providers=["CPUExecutionProvider"])
    return _sess, _tok


def embed(texts, batch=16):
    sess, tok = _load()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[i:i + batch])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = sess.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
        v = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.vstack(out).astype(np.float32)


def embed_cached(texts, name):
    """Embed with an on-disk cache (data/vectors/<name>.npy) keyed by the texts' hash,
    so a small server does not re-embed the corpus on every start."""
    import hashlib
    d = os.path.join(os.path.dirname(__file__), "..", "data", "vectors")
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    path = os.path.join(d, f"{name}-{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    v = embed(texts)
    for old in os.listdir(d):
        if old.startswith(name + "-"):
            os.remove(os.path.join(d, old))
    np.save(path, v)
    return v
