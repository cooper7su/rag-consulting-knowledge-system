#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

from src.retrieve import RetrievedChunk, Retriever


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe_docs(chunks: List[RetrievedChunk]) -> List[str]:
    seen = set()
    docs = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        docs.append(chunk.doc_id)
    return docs


def dcg(binary_relevances: List[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(binary_relevances))


def query_metrics(chunks: List[RetrievedChunk], expected_docs: set[str], top_k: int) -> dict:
    ranked_docs = dedupe_docs(chunks)[:top_k]
    hit = float(bool(expected_docs & set(ranked_docs)))

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in expected_docs:
            reciprocal_rank = 1.0 / rank
            break

    gains = [1 if doc_id in expected_docs else 0 for doc_id in ranked_docs]
    ideal = [1] * min(len(expected_docs), top_k)
    ndcg = dcg(gains) / dcg(ideal) if ideal else 0.0

    front_matter_hits = sum(
        1
        for chunk in chunks[:top_k]
        if any(flag.startswith("penalized:front_matter") or flag.startswith("filtered:") for flag in chunk.quality_flags)
    )

    return {
        "hit": hit,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "unique_docs": len(set(chunk.doc_id for chunk in chunks[:top_k])),
        "front_matter_hits": front_matter_hits,
        "ranked_docs": ranked_docs,
    }


def run_mode(retriever: Retriever, mode: str, questions: List[dict], top_k: int) -> dict:
    per_query = []
    aggregate = {
        "hit": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "unique_docs": 0.0,
        "front_matter_hits": 0.0,
        "judged": 0,
    }

    for row in questions:
        query = row["query"]
        expected_docs = set(row.get("expected_docs", []))
        chunks = retriever.vector_search(query, top_k=top_k) if mode == "vector" else retriever.hybrid_search(query, top_k=top_k)

        metrics = query_metrics(chunks, expected_docs, top_k=top_k)
        if expected_docs:
            aggregate["hit"] += metrics["hit"]
            aggregate["mrr"] += metrics["mrr"]
            aggregate["ndcg"] += metrics["ndcg"]
            aggregate["judged"] += 1

        aggregate["unique_docs"] += metrics["unique_docs"]
        aggregate["front_matter_hits"] += metrics["front_matter_hits"]

        per_query.append(
            {
                "query": query,
                "expected_docs": sorted(expected_docs),
                "metrics": metrics,
                "results": [
                    {
                        "rank": idx,
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "section": chunk.section,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "quality_flags": list(chunk.quality_flags),
                    }
                    for idx, chunk in enumerate(chunks, start=1)
                ],
            }
        )

    total_queries = len(questions)
    judged = max(aggregate["judged"], 1)
    summary = {
        "mode": mode,
        "queries": total_queries,
        "judged_queries": aggregate["judged"],
        f"hit@{top_k}": aggregate["hit"] / judged,
        "mrr": aggregate["mrr"] / judged,
        f"ndcg@{top_k}": aggregate["ndcg"] / judged,
        f"avg_unique_docs@{top_k}": aggregate["unique_docs"] / total_queries,
        f"avg_front_matter_hits@{top_k}": aggregate["front_matter_hits"] / total_queries,
    }
    return {"summary": summary, "per_query": per_query}


def print_mode_report(report: dict, top_k: int) -> None:
    summary = report["summary"]
    print(f"\n=== {summary['mode'].upper()} ===")
    for key, value in summary.items():
        if key == "mode":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPer-query results:")
    for row in report["per_query"]:
        print(f"- Query: {row['query']}")
        if row["expected_docs"]:
            print(f"  expected_docs: {row['expected_docs']}")
            print(
                f"  hit={row['metrics']['hit']:.0f} "
                f"mrr={row['metrics']['mrr']:.3f} "
                f"ndcg={row['metrics']['ndcg']:.3f}"
            )
        print(f"  ranked_docs: {row['metrics']['ranked_docs']}")
        for result in row["results"][:top_k]:
            flags = f" flags={result['quality_flags']}" if result["quality_flags"] else ""
            print(
                f"    [{result['rank']}] {result['doc_id']} "
                f"p{result['page_start']}-{result['page_end']} {result['section']}{flags}"
            )


def print_qualitative_analysis(reports: Dict[str, dict]) -> None:
    print("\n=== Qualitative Error Analysis ===")
    modes = list(reports.keys())
    base_mode = modes[0]
    base_queries = {row["query"]: row for row in reports[base_mode]["per_query"]}

    for query in base_queries:
        rows = {mode: next(item for item in reports[mode]["per_query"] if item["query"] == query) for mode in modes}
        mode_rankings = {mode: rows[mode]["metrics"]["ranked_docs"] for mode in modes}
        misses = [mode for mode in modes if rows[mode]["expected_docs"] and rows[mode]["metrics"]["hit"] == 0]
        noisy = [
            mode
            for mode in modes
            if rows[mode]["results"] and rows[mode]["results"][0]["quality_flags"]
        ]

        if not misses and len({tuple(ranking) for ranking in mode_rankings.values()}) == 1 and not noisy:
            continue

        print(f"- Query: {query}")
        for mode in modes:
            print(
                f"  {mode}: ranked_docs={rows[mode]['metrics']['ranked_docs']} "
                f"top1_flags={rows[mode]['results'][0]['quality_flags'] if rows[mode]['results'] else []}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index", help="Folder containing faiss.index / meta.parquet / manifest.json")
    ap.add_argument("--questions", required=True, help="JSONL file with query and expected_docs")
    ap.add_argument("--emb_model", default="local-hash-v1", help="Embedding model used for indexing")
    ap.add_argument("--modes", nargs="+", choices=["vector", "hybrid"], default=["hybrid"], help="Retrieval modes to compare")
    ap.add_argument("--top_k", type=int, default=5)
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

    questions = read_jsonl(Path(args.questions))
    if not questions:
        raise SystemExit("No evaluation questions found.")

    reports = {
        mode: run_mode(retriever, mode=mode, questions=questions, top_k=args.top_k)
        for mode in args.modes
    }

    for mode in args.modes:
        print_mode_report(reports[mode], top_k=args.top_k)

    if len(args.modes) > 1:
        print_qualitative_analysis(reports)


if __name__ == "__main__":
    main()
