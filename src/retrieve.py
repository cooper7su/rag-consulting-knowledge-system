#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Tuple

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    return [t for t in text.split() if t]


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    section: str
    page_start: int
    page_end: int
    score: float
    text: str


class Retriever:
    def __init__(self, index_path: str, meta_path: str, emb_model: str):
        self.index = faiss.read_index(index_path)
        self.meta = pd.read_parquet(meta_path)
        self.model = SentenceTransformer(emb_model)

        # BM25 corpus
        self.corpus_tokens = [_tokenize(t) for t in self.meta["text"].tolist()]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def vector_search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        out = []
        for i, s in zip(ids, scores):
            if i < 0:
                continue
            r = self.meta.iloc[int(i)]
            out.append(
                RetrievedChunk(
                    doc_id=r["doc_id"],
                    chunk_id=r["chunk_id"],
                    section=r["section"],
                    page_start=int(r["page_start"]),
                    page_end=int(r["page_end"]),
                    score=float(s),
                    text=r["text"],
                )
            )
        return out

    def hybrid_search(self, query: str, top_k: int = 8, bm25_k: int = 50, vec_k: int = 30) -> List[RetrievedChunk]:
        # BM25 candidates
        qtok = _tokenize(query)
        bm_scores = self.bm25.get_scores(qtok)
        bm_top = np.argsort(bm_scores)[::-1][:bm25_k].tolist()

        # Vector candidates
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        v_scores, v_ids = self.index.search(q, vec_k)
        v_ids = [int(x) for x in v_ids[0].tolist() if x >= 0]

        # Union candidates
        cand = list(dict.fromkeys(bm_top + v_ids))  # preserve order, unique

        # Vector re-rank over candidates
        cand_texts = self.meta.iloc[cand]["text"].tolist()
        cand_emb = self.model.encode(cand_texts, normalize_embeddings=True).astype("float32")
        qv = q[0]
        rerank_scores = cand_emb @ qv  # cosine via dot product

        order = np.argsort(rerank_scores)[::-1][:top_k]
        out = []
        for idx in order:
            faiss_id = cand[int(idx)]
            r = self.meta.iloc[int(faiss_id)]
            out.append(
                RetrievedChunk(
                    doc_id=r["doc_id"],
                    chunk_id=r["chunk_id"],
                    section=r["section"],
                    page_start=int(r["page_start"]),
                    page_end=int(r["page_end"]),
                    score=float(rerank_scores[int(idx)]),
                    text=r["text"],
                )
            )
        return out
