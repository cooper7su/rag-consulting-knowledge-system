#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.embeddings import load_embedder

INDEX_MANIFEST_VERSION = 2
REQUIRED_META_COLUMNS = {
    "doc_id",
    "chunk_id",
    "title",
    "source",
    "date",
    "file_name",
    "file_path",
    "section",
    "page_start",
    "page_end",
    "text",
}

HARD_FILTER_SECTIONS = {
    "TABLE OF CONTENTS",
    "CONTENTS",
}
SOFT_PENALTY_SECTIONS = {
    "PREFACE",
    "FOREWORD",
    "ACKNOWLEDGEMENTS",
    "ACKNOWLEDGMENTS",
}
DOT_LEADER_RE = re.compile(r"\.{4,}")


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    return [t for t in text.split() if t]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _minmax_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return {idx: 0.0 for idx in scores}
    return {idx: (value - lo) / (hi - lo) for idx, value in scores.items()}


@dataclass(frozen=True)
class RetrievalConfig:
    vector_candidate_k: int = 60
    hybrid_bm25_k: int = 80
    hybrid_vec_k: int = 60
    hybrid_dense_weight: float = 0.45
    hybrid_bm25_weight: float = 0.55
    short_text_filter_chars: int = 40
    short_text_penalty_chars: int = 120
    short_text_penalty: float = 0.12
    soft_section_penalty: float = 0.08
    same_doc_penalty: float = 0.12
    nearby_page_penalty: float = 0.05
    same_section_penalty: float = 0.02
    title_overlap_bonus: float = 0.08
    section_overlap_bonus: float = 0.05
    text_overlap_bonus: float = 0.10
    duplicate_similarity_threshold: float = 0.88


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    title: str
    source: str
    date: str | None
    file_name: str
    file_path: str
    section: str
    page_start: int
    page_end: int
    score: float
    text: str
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CandidateChunk:
    faiss_id: int
    row: pd.Series
    base_score: float
    vector_score: float
    bm25_score: float
    quality_flags: tuple[str, ...]
    filtered: bool


class Retriever:
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        emb_model: str,
        manifest_path: str | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.config = config or RetrievalConfig()
        self.index = faiss.read_index(index_path)
        self.meta = pd.read_parquet(meta_path)
        self.manifest = self._load_manifest(manifest_path)
        self._validate_index()
        model_to_load = self.manifest.get("effective_model", emb_model)
        self.embedder = load_embedder(model_to_load)

        self.corpus_tokens = [_tokenize(text) for text in self.meta["text"].tolist()]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    @staticmethod
    def _load_manifest(manifest_path: str | None) -> dict:
        if not manifest_path:
            return {}

        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Index manifest not found at {path}. Rebuild the index with the current pipeline."
            )

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_index(self) -> None:
        if self.manifest.get("index_manifest_version") != INDEX_MANIFEST_VERSION:
            raise ValueError("Index manifest version is missing or outdated. Rebuild the index.")

        for file_name in self.manifest.get("required_index_files", []):
            if not (self.index_path.parent / file_name).exists():
                raise FileNotFoundError(f"Index is incomplete. Missing required file: {file_name}")

        if int(self.manifest.get("chunk_schema_version", 0)) < 2:
            raise ValueError("Chunk schema version is outdated. Rebuild chunks and index.")

        missing_columns = REQUIRED_META_COLUMNS - set(self.meta.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Index metadata schema is outdated. Missing columns: {missing}")

        expected_chunks = self.manifest.get("num_chunks")
        if expected_chunks is not None and int(expected_chunks) != len(self.meta):
            raise ValueError("Index manifest chunk count does not match metadata rows.")

        if self.index.ntotal != len(self.meta):
            raise ValueError("FAISS index size does not match metadata row count.")

        manifest_columns = self.manifest.get("metadata_columns", [])
        if manifest_columns:
            missing_manifest_columns = set(manifest_columns) - set(self.meta.columns)
            if missing_manifest_columns:
                missing = ", ".join(sorted(missing_manifest_columns))
                raise ValueError(f"Index metadata is incomplete. Missing manifest columns: {missing}")

    def _assess_quality(self, row: pd.Series) -> tuple[bool, float, tuple[str, ...]]:
        flags: List[str] = []
        penalty = 0.0
        section_upper = str(row["section"]).strip().upper()
        text = str(row["text"])
        text_len = len(text.strip())

        if section_upper in HARD_FILTER_SECTIONS:
            flags.append("filtered:front_matter")
            return True, 0.0, tuple(flags)

        if DOT_LEADER_RE.search(text) and section_upper in {"PAGE CONTENT", "TABLE OF CONTENTS", "CONTENTS"}:
            flags.append("filtered:toc_pattern")
            return True, 0.0, tuple(flags)

        if text_len < self.config.short_text_filter_chars:
            flags.append("filtered:too_short")
            return True, 0.0, tuple(flags)

        if section_upper in SOFT_PENALTY_SECTIONS:
            flags.append("penalized:front_matter")
            penalty += self.config.soft_section_penalty

        if text_len < self.config.short_text_penalty_chars:
            flags.append("penalized:short_text")
            penalty += self.config.short_text_penalty

        return False, penalty, tuple(flags)

    def _metadata_bonus(self, row: pd.Series, query_tokens: set[str]) -> float:
        bonus = 0.0
        title_tokens = set(_tokenize(str(row["title"])))
        section_tokens = set(_tokenize(str(row["section"])))
        text_tokens = set(_tokenize(str(row["text"])))

        if query_tokens & title_tokens:
            bonus += self.config.title_overlap_bonus
        if query_tokens & section_tokens:
            bonus += self.config.section_overlap_bonus
        if query_tokens:
            overlap_ratio = len(query_tokens & text_tokens) / len(query_tokens)
            bonus += min(overlap_ratio, 1.0) * self.config.text_overlap_bonus
        return bonus

    def _build_candidates(
        self,
        candidate_ids: List[int],
        vector_scores: Dict[int, float],
        bm25_scores: Dict[int, float],
        query_tokens: set[str],
        mode: str,
    ) -> List[CandidateChunk]:
        unique_ids = list(dict.fromkeys(idx for idx in candidate_ids if idx >= 0))
        norm_vector = _minmax_normalize({idx: vector_scores.get(idx, 0.0) for idx in unique_ids})
        norm_bm25 = _minmax_normalize({idx: bm25_scores.get(idx, 0.0) for idx in unique_ids})

        candidates: List[CandidateChunk] = []
        for idx in unique_ids:
            row = self.meta.iloc[int(idx)]
            filtered, penalty, flags = self._assess_quality(row)
            bonus = self._metadata_bonus(row, query_tokens)

            if mode == "hybrid":
                base_score = (
                    self.config.hybrid_dense_weight * norm_vector.get(idx, 0.0)
                    + self.config.hybrid_bm25_weight * norm_bm25.get(idx, 0.0)
                )
            else:
                base_score = norm_vector.get(idx, 0.0)

            base_score += bonus
            base_score -= penalty

            candidates.append(
                CandidateChunk(
                    faiss_id=int(idx),
                    row=row,
                    base_score=float(base_score),
                    vector_score=float(vector_scores.get(idx, 0.0)),
                    bm25_score=float(bm25_scores.get(idx, 0.0)),
                    quality_flags=flags,
                    filtered=filtered,
                )
            )
        return candidates

    def _are_near_duplicates(self, cand: CandidateChunk, selected: CandidateChunk) -> bool:
        if cand.row["doc_id"] != selected.row["doc_id"]:
            return False

        if (
            int(cand.row["page_start"]) == int(selected.row["page_start"])
            and str(cand.row["section"]) == str(selected.row["section"])
        ):
            return True

        if abs(int(cand.row["page_start"]) - int(selected.row["page_start"])) > 1:
            return False

        text_a = _normalize_text(str(cand.row["text"]))[:600]
        text_b = _normalize_text(str(selected.row["text"]))[:600]
        if text_a == text_b:
            return True

        similarity = SequenceMatcher(None, text_a, text_b).ratio()
        return similarity >= self.config.duplicate_similarity_threshold

    def _diversified_select(self, candidates: List[CandidateChunk], top_k: int) -> List[RetrievedChunk]:
        remaining = [cand for cand in candidates if not cand.filtered]
        selected: List[CandidateChunk] = []

        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = float("-inf")

            for idx, cand in enumerate(remaining):
                if any(self._are_near_duplicates(cand, prev) for prev in selected):
                    continue

                adjusted = cand.base_score
                same_doc_count = sum(1 for prev in selected if prev.row["doc_id"] == cand.row["doc_id"])
                adjusted -= self.config.same_doc_penalty * same_doc_count

                nearby_page_count = sum(
                    1
                    for prev in selected
                    if prev.row["doc_id"] == cand.row["doc_id"]
                    and abs(int(prev.row["page_start"]) - int(cand.row["page_start"])) <= 1
                )
                adjusted -= self.config.nearby_page_penalty * nearby_page_count

                same_section_count = sum(
                    1
                    for prev in selected
                    if prev.row["doc_id"] == cand.row["doc_id"] and prev.row["section"] == cand.row["section"]
                )
                adjusted -= self.config.same_section_penalty * same_section_count

                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = idx

            if best_idx is None:
                break

            chosen = remaining.pop(best_idx)
            selected.append(
                CandidateChunk(
                    faiss_id=chosen.faiss_id,
                    row=chosen.row,
                    base_score=best_score,
                    vector_score=chosen.vector_score,
                    bm25_score=chosen.bm25_score,
                    quality_flags=chosen.quality_flags,
                    filtered=chosen.filtered,
                )
            )

        return [self._to_retrieved_chunk(cand) for cand in selected]

    @staticmethod
    def _to_retrieved_chunk(cand: CandidateChunk) -> RetrievedChunk:
        row = cand.row
        return RetrievedChunk(
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            title=row["title"],
            source=row["source"],
            date=row["date"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            section=row["section"],
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            score=float(cand.base_score),
            text=row["text"],
            quality_flags=cand.quality_flags,
        )

    def vector_search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        candidate_k = max(top_k * 6, self.config.vector_candidate_k)
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(query_vec, candidate_k)

        vector_scores = {
            int(idx): float(score)
            for idx, score in zip(ids[0].tolist(), scores[0].tolist())
            if idx >= 0
        }
        query_tokens = set(_tokenize(query))
        candidates = self._build_candidates(
            candidate_ids=list(vector_scores.keys()),
            vector_scores=vector_scores,
            bm25_scores={},
            query_tokens=query_tokens,
            mode="vector",
        )
        return self._diversified_select(candidates, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int = 8, bm25_k: int | None = None, vec_k: int | None = None) -> List[RetrievedChunk]:
        bm25_k = bm25_k or max(top_k * 8, self.config.hybrid_bm25_k)
        vec_k = vec_k or max(top_k * 6, self.config.hybrid_vec_k)

        query_tokens = _tokenize(query)
        bm_scores = self.bm25.get_scores(query_tokens)
        bm_top = np.argsort(bm_scores)[::-1][:bm25_k].tolist()
        bm25_scores = {int(idx): float(bm_scores[idx]) for idx in bm_top}

        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        dense_scores, dense_ids = self.index.search(query_vec, vec_k)
        vector_scores = {
            int(idx): float(score)
            for idx, score in zip(dense_ids[0].tolist(), dense_scores[0].tolist())
            if idx >= 0
        }

        candidate_ids = bm_top + list(vector_scores.keys())
        candidates = self._build_candidates(
            candidate_ids=candidate_ids,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            query_tokens=set(query_tokens),
            mode="hybrid",
        )
        return self._diversified_select(candidates, top_k=top_k)
