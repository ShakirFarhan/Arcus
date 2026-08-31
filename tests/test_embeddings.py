import numpy as np

from arcus.embeddings import embed


def test_embed_returns_normalized_vectors_of_expected_shape():
    vectors = embed(["hello world", "a completely different sentence"])

    assert vectors.shape[0] == 2
    # all-MiniLM-L6-v2's embedding dimension
    assert vectors.shape[1] == 384

    for vector in vectors:
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
