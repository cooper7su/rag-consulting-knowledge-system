#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm


@dataclass
class PageText:
    page: int
    text: str


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fix_hard_linebreaks(text: str) -> str:
    # Join broken lines in PDFs while keeping paragraph breaks.
    # Heuristic: merge single newlines within sentences; keep blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out = []
    buf = ""
    for line in lines:
        l = line.strip()
        if not l:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append("")  # paragraph break
            continue
        # If line ends with hyphen, merge without space
        if buf:
            if buf.endswith("-"):
                buf = buf[:-1] + l
            else:
                # If previous line seems incomplete, merge; else new paragraph line
                if re.search(r"[a-zA-Z0-9,;:]$", buf) and re.match(r"^[a-zA-Z0-9(]", l):
                    buf = buf + " " + l
                else:
                    buf = buf + " " + l
        else:
            buf = l
    if buf:
        out.append(buf.strip())
    # Rebuild with paragraph breaks
    rebuilt = "\n\n".join([p for p in out if p is not None])
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
    return rebuilt.strip()


def extract_pdf_pages(pdf_path: Path) -> List[PageText]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        txt = page.get_text("text")
        txt = normalize_whitespace(txt)
        txt = fix_hard_linebreaks(txt)
        pages.append(PageText(page=i + 1, text=txt))
    return pages


def detect_repeated_lines(pages: List[PageText], top_k: int = 2, bottom_k: int = 2, min_freq: float = 0.6) -> Tuple[set, set]:
    """
    Detect repeated header/footer lines by frequency across pages.
    """
    top_counts: Dict[str, int] = {}
    bottom_counts: Dict[str, int] = {}
    n = max(1, len(pages))

    for p in pages:
        lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
        top = lines[:top_k]
        bottom = lines[-bottom_k:] if len(lines) >= bottom_k else lines
        for ln in top:
            top_counts[ln] = top_counts.get(ln, 0) + 1
        for ln in bottom:
            bottom_counts[ln] = bottom_counts.get(ln, 0) + 1

    top_rep = {ln for ln, c in top_counts.items() if c / n >= min_freq and len(ln) <= 120}
    bottom_rep = {ln for ln, c in bottom_counts.items() if c / n >= min_freq and len(ln) <= 120}
    return top_rep, bottom_rep


def remove_headers_footers(text: str, top_rep: set, bottom_rep: set) -> str:
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        l = ln.strip()
        if l in top_rep or l in bottom_rep:
            continue
        cleaned.append(ln)
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


_heading_re = re.compile(
    r"^\s*(?:\d+(\.\d+){0,3}\s+)?[A-Z][A-Za-z0-9 /&(),\-]{3,}$"
)

def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """
    Very lightweight 'semantic chunking': split by heading-like lines.
    Returns list of (section_title, section_text).
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    sections = []
    current_title = "Introduction"
    buf: List[str] = []

    for ln in lines:
        l = ln.strip()
        if not l:
            buf.append("")
            continue
        # Heading heuristic
        if _heading_re.match(l) and len(l.split()) <= 12:
            # flush previous
            sec_text = "\n".join(buf).strip()
            if sec_text:
                sections.append((current_title, sec_text))
            current_title = l
            buf = []
        else:
            buf.append(ln)

    sec_text = "\n".join(buf).strip()
    if sec_text:
        sections.append((current_title, sec_text))
    return sections


def chunk_section_text(section_text: str, max_chars: int = 2200, overlap_chars: int = 250) -> List[str]:
    """
    Chunk by character length (robust across tokenizers).
    """
    t = re.sub(r"\n{3,}", "\n\n", section_text).strip()
    if len(t) <= max_chars:
        return [t]

    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        # try to cut at paragraph boundary
        cut = t.rfind("\n\n", start + int(max_chars * 0.6), end)
        if cut == -1 or cut <= start:
            cut = end
        chunk = t[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, cut - overlap_chars)
        if start == cut:  # safety
            start = cut
    return chunks


def build_chunks_for_doc(doc_id: str, pages: List[PageText]) -> List[dict]:
    top_rep, bottom_rep = detect_repeated_lines(pages)
    # merge all pages but keep page map
    per_page_clean = []
    for p in pages:
        cleaned = remove_headers_footers(p.text, top_rep, bottom_rep)
        per_page_clean.append(PageText(page=p.page, text=cleaned))

    # Create a big text with explicit page markers for traceability
    merged = []
    for p in per_page_clean:
        merged.append(f"\n\n[[PAGE:{p.page}]]\n\n{p.text}")
    merged_text = "\n".join(merged).strip()

    # Split into sections
    sections = split_by_headings(merged_text)

    chunks = []
    chunk_idx = 0

    # page marker parsing helpers
    page_marker = re.compile(r"\[\[PAGE:(\d+)\]\]")

    def pages_covered(txt: str) -> Tuple[int, int]:
        found = [int(x) for x in page_marker.findall(txt)]
        if not found:
            return (1, 1)
        return (min(found), max(found))

    for title, sec in sections:
        # remove markers from chunk text but use them to infer page span
        # keep markers in a copy to compute page span
        span_start, span_end = pages_covered(sec)
        sec_wo = page_marker.sub("", sec).strip()

        for c in chunk_section_text(sec_wo):
            chunk_idx += 1
            cid = f"{doc_id}_{chunk_idx:05d}"
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": cid,
                    "section": title,
                    "page_start": span_start,
                    "page_end": span_end,
                    "text": c,
                }
            )
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with PDF files")
    ap.add_argument("--output", required=True, help="Output chunks.jsonl path")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(list(in_dir.glob("*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {in_dir}")

    all_chunks = []
    for pdf in tqdm(pdfs, desc="Ingest PDFs"):
        doc_id = pdf.stem
        pages = extract_pdf_pages(pdf)
        chunks = build_chunks_for_doc(doc_id, pages)
        all_chunks.extend(chunks)

    with out_path.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(all_chunks)} chunks to: {out_path}")


if __name__ == "__main__":
    main()
