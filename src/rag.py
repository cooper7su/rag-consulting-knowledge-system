#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Literal

from src.retrieve import Retriever, RetrievedChunk
from src.compress import compress_chunk


def format_sources(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[S{i}] {c.doc_id} p{c.page_start}-{c.page_end} ({c.chunk_id})")
    return "\n".join(lines)


def build_context(query: str, chunks: List[RetrievedChunk], compress: bool = True) -> List[str]:
    ctx = []
    for i, c in enumerate(chunks, start=1):
        text = compress_chunk(query, c.text) if compress else c.text
        ctx.append(f"[S{i}] {c.doc_id} p{c.page_start}-{c.page_end} | {c.section}\n{text}")
    return ctx


def answer_extractive(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Consulting-style structured answer without needing an LLM.
    Uses compressed evidence snippets + clear citations.
    """
    ctx = build_context(query, chunks, compress=True)

    bullets = []
    for i, block in enumerate(ctx, start=1):
        # Take first ~1-2 sentences from compressed block as a "finding"
        finding = block.split("\n", 1)[-1].strip()
        finding = finding[:350].rstrip()
        bullets.append(f"- {finding} (see [S{i}])")

    out = []
    out.append("## Executive Summary")
    out.append("".join([]) or "\n".join(bullets[:3]))

    out.append("\n## Key Findings (evidence-backed)")
    out.append("\n".join(bullets))

    out.append("\n## Implications / Recommendations (draft)")
    out.append(
        "- Validate key assumptions with client context and latest data; prioritize high-impact levers aligned to strategy.\n"
        "- Use cited evidence to build a 1–2 page insight note (market, policy, risks, opportunities) and iterate with stakeholders.\n"
        "- If needed, extend the corpus and add domain-specific taxonomy for better coverage and consistency."
    )

    out.append("\n## Sources")
    out.append(format_sources(chunks))

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index", help="Folder containing faiss.index and meta.parquet")
    ap.add_argument("--emb_model", default="BAAI/bge-small-en-v1.5", help="Embedding model used for indexing")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    index_path = os.path.join(args.index_dir, "faiss.index")
    meta_path = os.path.join(args.index_dir, "meta.parquet")

    r = Retriever(index_path=index_path, meta_path=meta_path, emb_model=args.emb_model)

    if args.mode == "vector":
        chunks = r.vector_search(args.query, top_k=args.top_k)
    else:
        chunks = r.hybrid_search(args.query, top_k=args.top_k)

    print(answer_extractive(args.query, chunks))


if __name__ == "__main__":
    main()
