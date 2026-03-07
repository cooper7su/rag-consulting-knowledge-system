import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.retrieve import Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IndexPipelineTests(unittest.TestCase):
    def test_build_index_and_load_retriever_with_current_schema(self):
        row = {
            "chunk_schema_version": 2,
            "doc_id": "sample_doc",
            "title": "Sample Title",
            "title_source": "filename_fallback",
            "source": "Sample Source",
            "source_source": "pdf_author",
            "date": "2025-01-01",
            "date_source": "pdf_creation_date",
            "file_name": "sample.pdf",
            "file_path": "sample.pdf",
            "doc_total_pages": 3,
            "chunk_id": "sample_doc_00001",
            "section": "Overview",
            "page_start": 1,
            "page_end": 1,
            "text": "AI policy supports adoption through funding and governance.",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks_path = tmp_path / "chunks.jsonl"
            chunks_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            ingest_manifest = {
                "ingest_manifest_version": 2,
                "summary": {"total_chunks": 1},
                "documents": [],
                "skipped_files": [],
            }
            (tmp_path / "ingest_manifest.json").write_text(
                json.dumps(ingest_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            out_dir = tmp_path / "index"
            cmd = [
                sys.executable,
                "-m",
                "src.embed_index",
                "--chunks",
                str(chunks_path),
                "--out",
                str(out_dir),
                "--model",
                "local-hash-v1",
            ]
            subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            retriever = Retriever(
                index_path=str(out_dir / "faiss.index"),
                meta_path=str(out_dir / "meta.parquet"),
                manifest_path=str(out_dir / "manifest.json"),
                emb_model="local-hash-v1",
            )
            results = retriever.vector_search("AI policy adoption", top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].doc_id, "sample_doc")
            self.assertEqual(results[0].title, "Sample Title")


if __name__ == "__main__":
    unittest.main()
