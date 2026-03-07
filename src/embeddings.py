#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

import numpy as np


DEFAULT_LOCAL_MODEL = "local-hash-v1"
LOCAL_EMBED_DIM = 1536


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    return [tok for tok in text.split() if tok]


@dataclass
class EmbedderInfo:
    requested_model: str
    effective_model: str
    backend: str


class LocalHashEmbedder:
    def __init__(self, requested_model: str = DEFAULT_LOCAL_MODEL, dim: int = LOCAL_EMBED_DIM):
        self.dim = dim
        self.info = EmbedderInfo(
            requested_model=requested_model,
            effective_model=DEFAULT_LOCAL_MODEL,
            backend="local-hash",
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        embs = np.zeros((len(texts), self.dim), dtype="float32")

        for row_idx, text in enumerate(texts):
            for token in _tokenize(text):
                token_hash = hashlib.md5(token.encode("utf-8")).hexdigest()
                col_idx = int(token_hash, 16) % self.dim
                embs[row_idx, col_idx] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            np.divide(embs, np.clip(norms, 1e-12, None), out=embs)

        return embs


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.info = EmbedderInfo(
            requested_model=model_name,
            effective_model=model_name,
            backend="sentence-transformers",
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        ).astype("float32")


def load_embedder(model_name: str, allow_fallback: bool = True, verbose: bool = True):
    if model_name == DEFAULT_LOCAL_MODEL:
        return LocalHashEmbedder(requested_model=model_name)

    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:
        if not allow_fallback:
            raise
        if verbose:
            print(
                f"Warning: failed to load embedding model '{model_name}'. "
                f"Falling back to '{DEFAULT_LOCAL_MODEL}'. Reason: {exc}"
            )
        return LocalHashEmbedder(requested_model=model_name)
