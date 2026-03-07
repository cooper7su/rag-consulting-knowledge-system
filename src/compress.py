#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _keywords(q: str) -> set[str]:
    q = q.lower()
    q = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", q)
    toks = [t for t in q.split() if len(t) >= 2]
    return set(toks)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [sent.strip() for sent in _SENTENCE_BOUNDARY_RE.split(text) if sent.strip()]


def _score_sentence(query_terms: set[str], sentence: str) -> float:
    normalized = sentence.lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", normalized)
    toks = [t for t in normalized.split() if len(t) >= 2]
    if not toks:
        return 0.0

    tok_set = set(toks)
    overlap = len(query_terms & tok_set)
    density = overlap / max(len(tok_set), 1)
    return overlap + density


@dataclass
class EvidenceSpan:
    snippet: str
    sentence_start: int
    sentence_end: int
    match_score: float


def extract_evidence_span(
    query: str,
    text: str,
    max_sents: int = 3,
    context_window: int = 1,
    max_chars: int = 420,
) -> EvidenceSpan:
    query_terms = _keywords(query)
    sents = _split_sentences(text)
    if not sents:
        snippet = text.strip()[:max_chars].strip()
        return EvidenceSpan(snippet=snippet, sentence_start=0, sentence_end=0, match_score=0.0)

    scores = [_score_sentence(query_terms, sent) for sent in sents]
    best_idx = max(range(len(sents)), key=lambda idx: scores[idx])

    if scores[best_idx] <= 0:
        chosen = list(range(min(max_sents, len(sents))))
    else:
        chosen = [best_idx]
        neighbor_ids = [
            idx
            for idx in range(max(0, best_idx - context_window), min(len(sents), best_idx + context_window + 1))
            if idx != best_idx
        ]
        neighbor_ids.sort(key=lambda idx: (scores[idx], -abs(idx - best_idx)), reverse=True)
        for idx in neighbor_ids:
            if len(chosen) >= max_sents:
                break
            if scores[idx] > 0 or abs(idx - best_idx) == 1:
                chosen.append(idx)
        chosen.sort()

    snippet_parts: List[str] = []
    for idx in chosen:
        sentence = sents[idx]
        if sum(len(part) for part in snippet_parts) + len(sentence) > max_chars and snippet_parts:
            break
        snippet_parts.append(sentence)

    snippet = " ".join(snippet_parts).strip()
    if not snippet:
        snippet = " ".join(sents[: min(max_sents, len(sents))]).strip()[:max_chars].strip()

    return EvidenceSpan(
        snippet=snippet,
        sentence_start=chosen[0],
        sentence_end=chosen[-1],
        match_score=float(scores[best_idx]),
    )


def compress_chunk(query: str, text: str, max_sents: int = 5) -> str:
    return extract_evidence_span(query, text, max_sents=max_sents, context_window=1).snippet
