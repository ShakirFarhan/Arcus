import threading

import numpy as np

_model_cache = None
_model_lock = threading.Lock()


def get_embedding_model():
    # imported lazily so importing this module doesn't drag torch in for
    # callers that never actually need embeddings. cached as a singleton
    # once loaded since the load itself (not the import) is the slow
    # part, worth paying once per process, not once per call site.
    #
    # the lock matters once the CLI started warming this up on a
    # background thread (see cli.py): without it, a foreground call
    # landing while the warm-up thread is mid-load would see an empty
    # cache and kick off a second, redundant load instead of just
    # waiting for the first one to finish.
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    with _model_lock:
        if _model_cache is None:
            from sentence_transformers import SentenceTransformer

            _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def embed(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True)
