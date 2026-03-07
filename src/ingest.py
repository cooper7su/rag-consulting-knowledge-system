#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pypdf import PdfReader
from tqdm import tqdm

CHUNK_SCHEMA_VERSION = 2


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class PageFailure:
    page: int
    reason: str


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess (stdout/stderr captured)."""
    return subprocess.run(cmd, text=True, capture_output=True)


def pdf_num_pages(pdf_path: Path) -> int:
    res = run_cmd(["pdfinfo", str(pdf_path)])
    if res.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {pdf_path}\n{res.stderr}")
    m = re.search(r"Pages:\s+(\d+)", res.stdout)
    if not m:
        raise RuntimeError(f"Cannot parse page count from pdfinfo output for {pdf_path}")
    return int(m.group(1))


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_metadata_value(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def humanize_doc_id(doc_id: str) -> str:
    return re.sub(r"[_\-]+", " ", doc_id).strip() or doc_id


def normalize_doc_id(name: str) -> str:
    doc_id = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return doc_id or "document"


def dedupe_doc_id(doc_id: str, used_doc_ids: Dict[str, int]) -> str:
    count = used_doc_ids.get(doc_id, 0)
    used_doc_ids[doc_id] = count + 1
    if count == 0:
        return doc_id
    return f"{doc_id}_{count + 1}"


def choose_date(creation_date: object, modification_date: object) -> Tuple[Optional[str], str]:
    for value, source in (
        (creation_date, "pdf_creation_date"),
        (modification_date, "pdf_modification_date"),
    ):
        if isinstance(value, datetime):
            return value.date().isoformat(), source
        text = clean_metadata_value(value)
        if text:
            return text, source
    return None, "unavailable"


def collect_input_files(in_dir: Path) -> Tuple[List[Path], List[dict]]:
    pdfs: List[Path] = []
    skipped: List[dict] = []

    for path in sorted(p for p in in_dir.rglob("*") if p.is_file()):
        rel_path = path.relative_to(in_dir).as_posix()
        if any(part.startswith(".") for part in rel_path.split("/")):
            continue
        if path.suffix.lower() == ".pdf":
            pdfs.append(path)
            continue
        skipped.append(
            {
                "file_name": path.name,
                "file_path": rel_path,
                "reason": "unsupported_extension",
            }
        )

    return pdfs, skipped


def extract_document_metadata(pdf_path: Path, input_root: Path, used_doc_ids: Dict[str, int]) -> dict:
    rel_path = pdf_path.relative_to(input_root).as_posix()
    stem = normalize_doc_id(pdf_path.stem)
    doc_id = dedupe_doc_id(stem, used_doc_ids)

    raw_title = None
    raw_author = None
    raw_subject = None
    raw_creator = None
    raw_producer = None
    creation_date = None
    modification_date = None
    metadata_error = None

    try:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        raw_title = clean_metadata_value(getattr(meta, "title", None))
        raw_author = clean_metadata_value(getattr(meta, "author", None))
        raw_subject = clean_metadata_value(getattr(meta, "subject", None))
        raw_creator = clean_metadata_value(getattr(meta, "creator", None))
        raw_producer = clean_metadata_value(getattr(meta, "producer", None))
        creation_date = getattr(meta, "creation_date", None)
        modification_date = getattr(meta, "modification_date", None)
        total_pages = len(reader.pages)
    except Exception as exc:
        metadata_error = str(exc)
        total_pages = pdf_num_pages(pdf_path)

    title = raw_title or humanize_doc_id(doc_id)
    title_source = "pdf_title" if raw_title else "filename_fallback"

    source_candidates = [
        (raw_author, "pdf_author"),
        (raw_subject, "pdf_subject"),
        (raw_creator, "pdf_creator"),
        (raw_producer, "pdf_producer"),
    ]
    source = "unknown"
    source_source = "unavailable"
    for value, candidate_source in source_candidates:
        if value:
            source = value
            source_source = candidate_source
            break

    date, date_source = choose_date(creation_date, modification_date)

    return {
        "doc_id": doc_id,
        "title": title,
        "title_source": title_source,
        "source": source,
        "source_source": source_source,
        "date": date,
        "date_source": date_source,
        "file_name": pdf_path.name,
        "file_path": rel_path,
        "file_size_bytes": pdf_path.stat().st_size,
        "file_sha256": sha256_file(pdf_path),
        "total_pages": total_pages,
        "pdf_title_raw": raw_title,
        "pdf_author_raw": raw_author,
        "pdf_subject_raw": raw_subject,
        "pdf_creator_raw": raw_creator,
        "pdf_producer_raw": raw_producer,
        "metadata_error": metadata_error,
    }


def extract_page_text_poppler(
    pdf_path: Path,
    page: int,
    timeout_sec: int = 30,
) -> Tuple[bool, str, str]:
    """
    Extract one page via pdftotext into a temp file to avoid huge stdout capture.
    Returns (ok, text, err_msg).
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as tmp:
        tmp_path = tmp.name
        cmd = ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), tmp_path]

        try:
            res = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return False, "", f"pdftotext timeout (> {timeout_sec}s)"
        except Exception as e:
            return False, "", f"pdftotext exception: {e}"

        if res.returncode != 0:
            return False, "", f"pdftotext failed: {res.stderr.strip()[:500]}"

        try:
            size_bytes = Path(tmp_path).stat().st_size
        except Exception as e:
            return False, "", f"stat temp text failed: {e}"

        # ✅ 防爆：临时文本太大直接跳过（避免 read_text 触发内存峰值）
        # 先用 5MB；如果还想更保守改成 2MB
        MAX_TMP_BYTES = 5 * 1024 * 1024

        if size_bytes > MAX_TMP_BYTES:
            return False, "", f"temp_text_too_large({size_bytes} bytes > {MAX_TMP_BYTES})"

        try:
            text = Path(tmp_path).read_text(errors="ignore")
        except Exception as e:
            return False, "", f"read temp text failed: {e}"

        return True, text, ""



def normalize_whitespace(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fix_hard_linebreaks(text: str) -> str:
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
            out.append("")
            continue
        if buf:
            if buf.endswith("-"):
                buf = buf[:-1] + l
            else:
                buf = buf + " " + l
        else:
            buf = l
    if buf:
        out.append(buf.strip())
    rebuilt = "\n\n".join([p for p in out if p is not None])
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
    return rebuilt.strip()


def extract_pdf_pages(
    pdf_path: Path,
    page_start: int = 1,
    page_end: Optional[int] = None,
    skip_bad_pages: bool = True,
    fail_log_path: Optional[Path] = None,
    timeout_sec: int = 30,
    max_page_chars: int = 200_000,
) -> Tuple[List[PageText], List[PageFailure], int]:
    """
    Extract pages in [page_start, page_end] (inclusive). 1-based.
    If skip_bad_pages=True, failures are logged and skipped.
    """
    total = pdf_num_pages(pdf_path)
    start = max(1, page_start)
    end = page_end if (page_end is not None and page_end > 0) else total
    end = min(end, total)

    if start > end:
        raise ValueError(f"Invalid page range: start={start} > end={end} for {pdf_path.name}")

    pages: List[PageText] = []
    failures: List[PageFailure] = []
    for p in range(start, end + 1):
        ok, raw, err = extract_page_text_poppler(pdf_path, p, timeout_sec=timeout_sec)
        if not ok:
            failures.append(PageFailure(page=p, reason=err))
            if fail_log_path:
                fail_log_path.parent.mkdir(parents=True, exist_ok=True)
                with fail_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"doc": pdf_path.name, "page": p, "reason": err},
                        ensure_ascii=False
                    ) + "\n")
            if skip_bad_pages:
                continue
            raise RuntimeError(f"{pdf_path.name} page {p} extract failed: {err}")

        # Guard: extremely large page text can cause memory spikes downstream
        if max_page_chars and len(raw) > max_page_chars:
            reason = f"page_text_too_large(len={len(raw)}>{max_page_chars})"
            failures.append(PageFailure(page=p, reason=reason))
            if fail_log_path:
                fail_log_path.parent.mkdir(parents=True, exist_ok=True)
                with fail_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"doc": pdf_path.name, "page": p, "reason": reason},
                        ensure_ascii=False
                    ) + "\n")
            if skip_bad_pages:
                continue
            raise RuntimeError(f"{pdf_path.name} page {p}: {reason}")

        txt = normalize_whitespace(raw)
        txt = fix_hard_linebreaks(txt)

        # Guard: empty/near-empty page
        if len(txt.strip()) < 10:
            reason = "page_text_too_short"
            if skip_bad_pages:
                failures.append(PageFailure(page=p, reason=reason))
                if fail_log_path:
                    fail_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with fail_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(
                            {"doc": pdf_path.name, "page": p, "reason": reason},
                            ensure_ascii=False
                        ) + "\n")
                continue
            # else keep it (rarely useful)
        pages.append(PageText(page=p, text=txt))

    return pages, failures, total


def detect_repeated_lines(
    pages: List[PageText],
    top_k: int = 2,
    bottom_k: int = 2,
    min_freq: float = 0.6,
) -> Tuple[set, set]:
    """
    Detect repeated header/footer lines by frequency across pages.
    Disabled for small page counts to avoid deleting real content.
    """
    if len(pages) < 6:
        return set(), set()

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

    top_rep = {ln for ln, c in top_counts.items() if c / n >= min_freq and len(ln) <= 160}
    bottom_rep = {ln for ln, c in bottom_counts.items() if c / n >= min_freq and len(ln) <= 160}
    return top_rep, bottom_rep


def remove_headers_footers(text: str, top_rep: set, bottom_rep: set) -> str:
    if not (top_rep or bottom_rep):
        return re.sub(r"\n{3,}", "\n\n", text).strip()

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


_heading_re = re.compile(r"^\s*(?:\d+(\.\d+){0,3}\s+)?[A-Z][A-Za-z0-9 /&(),\-]{3,}$")


def split_by_headings(text: str) -> List[Tuple[str, str]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    sections = []
    current_title = "Page Content"
    buf: List[str] = []

    for ln in lines:
        l = ln.strip()
        if not l:
            buf.append("")
            continue
        if _heading_re.match(l) and len(l.split()) <= 14:
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


def chunk_text(text: str, max_chars: int = 2200, overlap_chars: int = 250) -> List[str]:
    t = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(t) <= max_chars:
        return [t]

    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        cut = t.rfind("\n\n", start + int(max_chars * 0.6), end)
        if cut == -1 or cut <= start:
            cut = end
        chunk = t[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        if cut >= len(t):
            break

        next_start = max(start + 1, cut - overlap_chars)
        if next_start <= start:
            next_start = cut
        start = next_start
    return chunks


def build_chunks_for_doc(doc_meta: dict, pages: List[PageText]) -> List[dict]:
    doc_id = doc_meta["doc_id"]
    top_rep, bottom_rep = detect_repeated_lines(pages)
    chunks: List[dict] = []
    chunk_idx = 0

    for p in pages:
        cleaned = remove_headers_footers(p.text, top_rep, bottom_rep)

        if not cleaned or len(cleaned) < 20:
            continue

        sections = split_by_headings(cleaned)
        for title, sec in sections:
            for c in chunk_text(sec):
                if len(c.strip()) < 20:
                    continue
                chunk_idx += 1
                cid = f"{doc_id}_{chunk_idx:05d}"
                chunks.append(
                    {
                        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                        "doc_id": doc_id,
                        "title": doc_meta["title"],
                        "title_source": doc_meta["title_source"],
                        "source": doc_meta["source"],
                        "source_source": doc_meta["source_source"],
                        "date": doc_meta["date"],
                        "date_source": doc_meta["date_source"],
                        "file_name": doc_meta["file_name"],
                        "file_path": doc_meta["file_path"],
                        "doc_total_pages": doc_meta["total_pages"],
                        "chunk_id": cid,
                        "section": title,
                        "page_start": p.page,
                        "page_end": p.page,
                        "text": c,
                    }
                )
    return chunks


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest_path_info(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    return {
        "path": path.as_posix(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with PDF files")
    ap.add_argument("--output", required=True, help="Output chunks.jsonl path")

    # Page range (1-based, inclusive)
    ap.add_argument("--page_start", type=int, default=1, help="Start page (1-based, inclusive)")
    ap.add_argument("--page_end", type=int, default=0, help="End page (inclusive). 0 = till end")

    # Robustness
    ap.add_argument("--skip_bad_pages", action="store_true", help="Skip pages that fail extraction")
    ap.add_argument("--fail_log", default="", help="Write skipped/failed pages to JSONL file")
    ap.add_argument("--timeout_sec", type=int, default=30, help="pdftotext timeout per page")
    ap.add_argument("--max_page_chars", type=int, default=200000, help="Skip pages with extracted text larger than this")
    ap.add_argument("--manifest_out", default="", help="Write document-level ingest manifest JSON")
    ap.add_argument("--documents_out", default="", help="Write per-document processing report JSONL")

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fail_log_path = Path(args.fail_log) if args.fail_log else None
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_path.with_name("ingest_manifest.json")
    documents_path = Path(args.documents_out) if args.documents_out else out_path.with_name("documents.jsonl")

    pdfs, skipped_files = collect_input_files(in_dir)
    if not pdfs:
        raise SystemExit(f"No PDF files found in {in_dir}")

    all_chunks: List[dict] = []
    doc_reports: List[dict] = []
    used_doc_ids: Dict[str, int] = {}
    for pdf in tqdm(pdfs, desc="Ingest PDFs"):
        doc_meta = extract_document_metadata(pdf, in_dir, used_doc_ids)

        try:
            pages, failed_pages, total_pages = extract_pdf_pages(
                pdf,
                page_start=args.page_start,
                page_end=(args.page_end if args.page_end > 0 else None),
                skip_bad_pages=args.skip_bad_pages,
                fail_log_path=fail_log_path,
                timeout_sec=args.timeout_sec,
                max_page_chars=args.max_page_chars,
            )
            doc_meta["total_pages"] = total_pages
            chunks = build_chunks_for_doc(doc_meta, pages)
            all_chunks.extend(chunks)

            doc_reports.append(
                {
                    **doc_meta,
                    "status": "included" if chunks else "included_no_chunks",
                    "pages_extracted": len(pages),
                    "failed_pages": [failure.__dict__ for failure in failed_pages],
                    "failed_page_count": len(failed_pages),
                    "chunk_count": len(chunks),
                }
            )
        except Exception as exc:
            doc_reports.append(
                {
                    **doc_meta,
                    "status": "failed",
                    "pages_extracted": 0,
                    "failed_pages": [],
                    "failed_page_count": 0,
                    "chunk_count": 0,
                    "error": str(exc),
                }
            )

    write_jsonl(out_path, all_chunks)
    write_jsonl(documents_path, doc_reports)

    summary = {
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "input_dir": manifest_path_info(in_dir),
        "output_chunks": manifest_path_info(out_path),
        "documents_report": manifest_path_info(documents_path),
        "fail_log": manifest_path_info(fail_log_path),
        "scanned_files": len(pdfs) + len(skipped_files),
        "pdf_files_discovered": len(pdfs),
        "included_documents": sum(1 for report in doc_reports if report["status"] != "failed"),
        "documents_with_chunks": sum(1 for report in doc_reports if report["chunk_count"] > 0),
        "documents_without_chunks": sum(1 for report in doc_reports if report["status"] == "included_no_chunks"),
        "failed_documents": sum(1 for report in doc_reports if report["status"] == "failed"),
        "skipped_files": len(skipped_files),
        "total_chunks": len(all_chunks),
        "total_failed_pages": sum(report["failed_page_count"] for report in doc_reports),
    }
    manifest = {
        "ingest_manifest_version": 2,
        "summary": summary,
        "documents": doc_reports,
        "skipped_files": skipped_files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote {len(all_chunks)} chunks to: {out_path}")
    print(f"✅ Wrote {len(doc_reports)} document reports to: {documents_path}")
    print(f"✅ Wrote ingest manifest to: {manifest_path}")
    if fail_log_path:
        print(f"📝 Failed/skipped pages logged to: {fail_log_path}")


if __name__ == "__main__":
    main()
