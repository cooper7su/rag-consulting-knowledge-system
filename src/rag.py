#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List

from src.compress import extract_evidence_span
from src.retrieve import RetrievedChunk, Retriever


def _format_page_range(chunk: RetrievedChunk) -> str:
    if chunk.page_start == chunk.page_end:
        return f"p.{chunk.page_start}"
    return f"pp.{chunk.page_start}-{chunk.page_end}"


def _trim_snippet(snippet: str, max_chars: int = 260) -> str:
    snippet = snippet.strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def format_sources(chunks: List[RetrievedChunk], snippets: List[str]) -> str:
    lines = []
    for i, (chunk, snippet) in enumerate(zip(chunks, snippets), start=1):
        lines.append(
            f"[S{i}] {chunk.title} | {chunk.section} | {_format_page_range(chunk)} | "
            f"\"{_trim_snippet(snippet, max_chars=220)}\""
        )
    return "\n".join(lines)


def build_evidence(query: str, chunks: List[RetrievedChunk]) -> List[dict]:
    evidence = []
    for i, chunk in enumerate(chunks, start=1):
        span = extract_evidence_span(query, chunk.text, max_sents=3, context_window=1, max_chars=420)
        evidence.append(
            {
                "label": f"S{i}",
                "chunk": chunk,
                "snippet": span.snippet,
                "score": span.match_score,
            }
        )
    return evidence


def answer_extractive(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Consulting-style structured answer without needing an LLM.
    Uses narrower evidence snippets and cleaner source formatting.
    """
    evidence = build_evidence(query, chunks)
    snippets = [item["snippet"] for item in evidence]

    bullets = []
    for item in evidence:
        snippet = _trim_snippet(item["snippet"], max_chars=320)
        bullets.append(f"- {snippet} (see [{item['label']}])")

    out = []
    out.append("## Executive Summary")
    out.append("\n".join(bullets[:3]))

    out.append("\n## Key Findings (evidence-backed)")
    out.append("\n".join(bullets))

    out.append("\n## Implications / Recommendations (draft)")
    out.append(
        "- Validate the retrieved evidence against the client question and decision context before drawing conclusions.\n"
        "- Use the cited snippets to assemble a short research memo covering policy direction, implementation risks, and strategic implications.\n"
        "- If the evidence is too concentrated in one source, broaden the corpus or refine the query before treating the answer as representative."
    )

    out.append("\n## Sources")
    out.append(format_sources(chunks, snippets))

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index", help="Folder containing faiss.index and meta.parquet")
    ap.add_argument("--emb_model", default="local-hash-v1", help="Embedding model used for indexing")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    index_path = os.path.join(args.index_dir, "faiss.index")
    meta_path = os.path.join(args.index_dir, "meta.parquet")
    manifest_path = os.path.join(args.index_dir, "manifest.json")

    retriever = Retriever(
        index_path=index_path,
        meta_path=meta_path,
        emb_model=args.emb_model,
        manifest_path=manifest_path,
    )

    if args.mode == "vector":
        chunks = retriever.vector_search(args.query, top_k=args.top_k)
    else:
        chunks = retriever.hybrid_search(args.query, top_k=args.top_k)

    print(answer_extractive(args.query, chunks))


if __name__ == "__main__":
    main()
