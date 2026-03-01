import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.global_retrieval import GlobalRetriever
from retrieval.spatial_retrieval import SpatialRetriever
from retrieval.combined_retrieval import CombinedRetriever


def test_global_retriever_basic():
    retriever = GlobalRetriever(similarity_metric="cosine", exclude_query=True, exclude_test_set=True)

    ids = ["ref_0", "ref_1", "ref_2", "ref_3", "ref_4", "test_0"]
    embeddings = np.random.RandomState(42).randn(6, 128).astype(np.float32)
    labels = [0, 1, 0, 1, 0, 1]
    splits = ["reference", "reference", "reference", "reference", "reference", "test"]

    retriever.build_index(ids, embeddings, labels, splits)

    result = retriever.retrieve(
        query_id="test_0",
        query_embedding=embeddings[5],
        k=3,
        encoder_name="test_encoder",
        encoder_version="v1",
        preprocessing_hash="abc123",
    )

    assert len(result.neighbor_ids) == 3
    assert "test_0" not in result.neighbor_ids
    assert all(nid.startswith("ref_") for nid in result.neighbor_ids)
    assert len(result.neighbor_scores) == 3
    assert result.neighbor_scores == sorted(result.neighbor_scores, reverse=True)
    assert result.method == "global"
    assert result.encoder_name == "test_encoder"
    assert result.query_embedding_hash != ""
    print("PASS: test_global_retriever_basic")


def test_global_retriever_excludes_query():
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=False)

    ids = ["a", "b", "c"]
    embeddings = np.eye(3, dtype=np.float32)
    labels = [0, 1, 0]
    splits = ["test", "test", "test"]

    retriever.build_index(ids, embeddings, labels, splits)

    result = retriever.retrieve(query_id="a", query_embedding=embeddings[0], k=2)
    assert "a" not in result.neighbor_ids
    assert len(result.neighbor_ids) == 2
    print("PASS: test_global_retriever_excludes_query")


def test_global_retriever_cosine_correctness():
    retriever = GlobalRetriever(exclude_query=False, exclude_test_set=False)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ids = ["q", "near", "mid", "far"]
    labels = [0, 0, 0, 0]
    splits = ["reference"] * 4

    retriever.build_index(ids, embeddings, labels, splits)
    result = retriever.retrieve(query_id="q", query_embedding=embeddings[0], k=4)
    assert result.neighbor_ids[0] == "q"
    assert result.neighbor_ids[1] == "near"
    print("PASS: test_global_retriever_cosine_correctness")


def test_spatial_retriever_basic():
    retriever = SpatialRetriever(exclude_query=True, exclude_test_set=True)

    rng = np.random.RandomState(42)
    n_patches = 16
    dim = 64
    ids = ["ref_0", "ref_1", "ref_2", "test_0"]
    spatial_features = [rng.randn(n_patches, dim).astype(np.float32) for _ in range(4)]
    labels = [0, 1, 0, 1]
    splits = ["reference", "reference", "reference", "test"]

    retriever.build_index(ids, spatial_features, labels, splits)

    result = retriever.retrieve(
        query_id="test_0",
        query_spatial=spatial_features[3],
        k=2,
    )

    assert len(result.neighbor_ids) == 2
    assert "test_0" not in result.neighbor_ids
    assert result.method == "spatial"
    print("PASS: test_spatial_retriever_basic")


def test_combined_retriever_basic():
    retriever = CombinedRetriever(alpha=0.5, exclude_query=True, exclude_test_set=True)

    rng = np.random.RandomState(42)
    dim_g = 128
    n_patches = 16
    dim_s = 64
    n = 5

    ids = [f"ref_{i}" for i in range(n - 1)] + ["test_0"]
    global_emb = rng.randn(n, dim_g).astype(np.float32)
    spatial_feats = [rng.randn(n_patches, dim_s).astype(np.float32) for _ in range(n)]
    labels = [0, 1, 0, 1, 0]
    splits = ["reference"] * (n - 1) + ["test"]

    retriever.build_index(ids, global_emb, spatial_feats, labels, splits)

    result = retriever.retrieve(
        query_id="test_0",
        query_global=global_emb[4],
        query_spatial=spatial_feats[4],
        k=3,
    )

    assert len(result.neighbor_ids) == 3
    assert "test_0" not in result.neighbor_ids
    assert result.method == "global_spatial"
    assert result.neighbor_scores == sorted(result.neighbor_scores, reverse=True)
    print("PASS: test_combined_retriever_basic")


def test_combined_retriever_alpha_extremes():
    rng = np.random.RandomState(123)
    dim_g = 32
    n_patches = 4
    dim_s = 16
    n = 6

    ids = [f"ref_{i}" for i in range(n - 1)] + ["test_0"]
    global_emb = rng.randn(n, dim_g).astype(np.float32)
    spatial_feats = [rng.randn(n_patches, dim_s).astype(np.float32) for _ in range(n)]
    labels = list(range(n))
    splits = ["reference"] * (n - 1) + ["test"]

    retriever_global = CombinedRetriever(alpha=1.0, exclude_query=True, exclude_test_set=True)
    retriever_global.build_index(ids, global_emb, spatial_feats, labels, splits)
    result_g = retriever_global.retrieve(
        query_id="test_0", query_global=global_emb[-1], query_spatial=spatial_feats[-1], k=3)

    pure_global = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    pure_global.build_index(ids, global_emb, labels, splits)
    result_pg = pure_global.retrieve(query_id="test_0", query_embedding=global_emb[-1], k=3)

    assert result_g.neighbor_ids == result_pg.neighbor_ids
    print("PASS: test_combined_retriever_alpha_extremes")


def test_retriever_batch():
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    rng = np.random.RandomState(42)
    n_ref = 20
    n_test = 5
    dim = 64

    ids = [f"ref_{i}" for i in range(n_ref)] + [f"test_{i}" for i in range(n_test)]
    embeddings = rng.randn(n_ref + n_test, dim).astype(np.float32)
    labels = [0] * n_ref + [1] * n_test
    splits = ["reference"] * n_ref + ["test"] * n_test

    retriever.build_index(ids, embeddings, labels, splits)

    query_ids = [f"test_{i}" for i in range(n_test)]
    query_embs = embeddings[n_ref:]

    results = retriever.retrieve_batch(query_ids, query_embs, k=4)
    assert len(results) == n_test
    for r in results:
        assert len(r.neighbor_ids) == 4
        assert all(nid.startswith("ref_") for nid in r.neighbor_ids)
    print("PASS: test_retriever_batch")


if __name__ == "__main__":
    test_global_retriever_basic()
    test_global_retriever_excludes_query()
    test_global_retriever_cosine_correctness()
    test_spatial_retriever_basic()
    test_combined_retriever_basic()
    test_combined_retriever_alpha_extremes()
    test_retriever_batch()
    print("\nAll retrieval tests passed.")
