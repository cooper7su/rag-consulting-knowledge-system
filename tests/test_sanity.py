import unittest

import numpy as np

from src.compress import compress_chunk
from src.embeddings import DEFAULT_LOCAL_MODEL, LOCAL_EMBED_DIM, load_embedder
from src.ingest import chunk_text


class SanityTests(unittest.TestCase):
    def test_chunk_text_finishes_for_long_inputs(self):
        chunks = chunk_text("A" * 4453)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertLessEqual(len(chunks), 5)
        self.assertTrue(all(chunks))
        self.assertLessEqual(max(len(chunk) for chunk in chunks), 2200)

    def test_compress_chunk_works_without_nltk_data(self):
        text = "This is a test sentence. Another policy sentence about AI adoption. Final sentence."
        compressed = compress_chunk("AI policy impacts", text)
        self.assertIn("policy sentence about AI adoption", compressed)

    def test_local_hash_embedder_is_deterministic(self):
        embedder = load_embedder(DEFAULT_LOCAL_MODEL, verbose=False)
        emb_a = embedder.encode(["AI policy and adoption"])
        emb_b = embedder.encode(["AI policy and adoption"])

        self.assertEqual(emb_a.shape, (1, LOCAL_EMBED_DIM))
        self.assertTrue(np.allclose(emb_a, emb_b))
        self.assertAlmostEqual(float(np.linalg.norm(emb_a[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
