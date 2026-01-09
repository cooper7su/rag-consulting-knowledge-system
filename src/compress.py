#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import List

import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize


def _keywords(q: str) -> set:
    q = q.lower()
    q = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", q)
    toks = [t for t in q.split() if len(t) >= 2]
    return set(toks)


def compress_chunk(query: str, text: str, max_sents: int = 5) -> str:
    """
    Extract the most query-relevant sentences from a chunk.
    """
    kws = _keywords(query)
    sents = sent_tokenize(text)
    if not sents:
        return text

    scored = []
    for s in sents:
        st = s.lower()
        st = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", st)
        toks = set([t for t in st.split() if len(t) >= 2])
        overlap = len(kws & toks)
        scored.append((overlap, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for sc, s in scored[:max_sents] if sc > 0]
    if not best:
        # fallback: first few sentences
        best = sents[: min(max_sents, len(sents))]
    return " ".join(best).strip()
