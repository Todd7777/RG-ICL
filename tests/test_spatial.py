import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.spatial_retrieval import SpatialRetriever
from retrieval.combined_retrieval import CombinedRetriever


def test_spatial_self_similarity_is_maximal():
    retriever = SpatialRetriever(exclude_query=False, exclude_test_set=False)

    rng = np.random.RandomState(42)
    n_patches = 16
    dim = 64
    n = 5

    ids = [f"img_{i}" for i in range(n)]
    spatial_features = [rng.randn(n_patches, dim).astype(np.float32) for _ in range(n)]
    labels = list(range(n))
    splits = ["reference"] * n

    retriever.build_index(ids, spatial_features, labels, splits)

    for i in range(n):
        result = retriever.retrieve(query_id=ids[i], query_spatial=spatial_features[i], k=1)
        assert result.neighbor_ids[0] == ids[i], \
            f"Self should be most similar, got {result.neighbor_ids[0]} for {ids[i]}"

    print("PASS: test_spatial_self_similarity_is_maximal")


def test_spatial_identical_patches_high_similarity():
    retriever = SpatialRetriever(exclude_query=True, exclude_test_set=False)

    rng = np.random.RandomState(42)
    n_patches = 16
    dim = 64

    base = rng.randn(n_patches, dim).astype(np.float32)
    similar = base + rng.randn(n_patches, dim).astype(np.float32) * 0.01
    different = rng.randn(n_patches, dim).astype(np.float32) * 10

    ids = ["query", "similar", "different"]
    spatial_features = [base, similar, different]
    labels = [0, 0, 1]
    splits = ["test", "reference", "reference"]

    retriever.build_index(ids, spatial_features, labels, splits)

    result = retriever.retrieve(query_id="query", query_spatial=base, k=2)
    assert result.neighbor_ids[0] == "similar"
    print("PASS: test_spatial_identical_patches_high_similarity")


def test_spatial_different_patch_counts():
    retriever = SpatialRetriever(exclude_query=True, exclude_test_set=False)

    rng = np.random.RandomState(42)
    dim = 32

    query_patches = rng.randn(16, dim).astype(np.float32)
    ref1_patches = rng.randn(25, dim).astype(np.float32)
    ref2_patches = rng.randn(9, dim).astype(np.float32)

    ids = ["query", "ref1", "ref2"]
    spatial_features = [query_patches, ref1_patches, ref2_patches]
    labels = [0, 0, 1]
    splits = ["test", "reference", "reference"]

    retriever.build_index(ids, spatial_features, labels, splits)
    result = retriever.retrieve(query_id="query", query_spatial=query_patches, k=2)

    assert len(result.neighbor_ids) == 2
    assert all(isinstance(s, float) for s in result.neighbor_scores)
    print("PASS: test_spatial_different_patch_counts")


def test_spatial_deterministic():
    rng = np.random.RandomState(42)
    n_patches = 16
    dim = 64
    n = 10

    ids = [f"img_{i}" for i in range(n)]
    spatial_features = [rng.randn(n_patches, dim).astype(np.float32) for _ in range(n)]
    labels = list(range(n))
    splits = ["reference"] * n

    r1 = SpatialRetriever(exclude_query=True, exclude_test_set=False)
    r1.build_index(ids, spatial_features, labels, splits)
    res1 = r1.retrieve(query_id="img_0", query_spatial=spatial_features[0], k=3)

    r2 = SpatialRetriever(exclude_query=True, exclude_test_set=False)
    r2.build_index(ids, spatial_features, labels, splits)
    res2 = r2.retrieve(query_id="img_0", query_spatial=spatial_features[0], k=3)

    assert res1.neighbor_ids == res2.neighbor_ids
    assert res1.neighbor_scores == res2.neighbor_scores
    print("PASS: test_spatial_deterministic")


def test_combined_spatial_component_matters():
    rng = np.random.RandomState(42)
    dim_g = 32
    n_patches = 8
    dim_s = 16
    n = 5

    ids = ["query", "global_near", "spatial_near", "both_far1", "both_far2"]
    splits = ["test", "reference", "reference", "reference", "reference"]
    labels = [0, 0, 0, 1, 1]

    global_emb = np.zeros((n, dim_g), dtype=np.float32)
    global_emb[0] = rng.randn(dim_g)
    global_emb[1] = global_emb[0] + rng.randn(dim_g) * 0.01
    global_emb[2] = rng.randn(dim_g) * 5
    global_emb[3] = rng.randn(dim_g) * 5
    global_emb[4] = rng.randn(dim_g) * 5

    spatial_feats = [rng.randn(n_patches, dim_s).astype(np.float32) for _ in range(n)]
    spatial_feats[2] = spatial_feats[0] + rng.randn(n_patches, dim_s).astype(np.float32) * 0.01

    ret_global = CombinedRetriever(alpha=1.0, exclude_query=True, exclude_test_set=True)
    ret_global.build_index(ids, global_emb, spatial_feats, labels, splits)
    res_g = ret_global.retrieve(query_id="query", query_global=global_emb[0],
                                 query_spatial=spatial_feats[0], k=1)
    assert res_g.neighbor_ids[0] == "global_near"

    ret_spatial = CombinedRetriever(alpha=0.0, exclude_query=True, exclude_test_set=True)
    ret_spatial.build_index(ids, global_emb, spatial_feats, labels, splits)
    res_s = ret_spatial.retrieve(query_id="query", query_global=global_emb[0],
                                  query_spatial=spatial_feats[0], k=1)
    assert res_s.neighbor_ids[0] == "spatial_near"

    print("PASS: test_combined_spatial_component_matters")


def test_spatial_score_range():
    retriever = SpatialRetriever(exclude_query=False, exclude_test_set=False)

    rng = np.random.RandomState(42)
    n_patches = 16
    dim = 64
    n = 5

    ids = [f"img_{i}" for i in range(n)]
    spatial_features = [rng.randn(n_patches, dim).astype(np.float32) for _ in range(n)]
    labels = list(range(n))
    splits = ["reference"] * n

    retriever.build_index(ids, spatial_features, labels, splits)
    result = retriever.retrieve(query_id="img_0", query_spatial=spatial_features[0], k=n)

    for score in result.neighbor_scores:
        assert -1.0 <= score <= 1.0, f"Score {score} out of [-1, 1] range"

    print("PASS: test_spatial_score_range")


if __name__ == "__main__":
    test_spatial_self_similarity_is_maximal()
    test_spatial_identical_patches_high_similarity()
    test_spatial_different_patch_counts()
    test_spatial_deterministic()
    test_combined_spatial_component_matters()
    test_spatial_score_range()
    print("\nAll spatial tests passed.")
