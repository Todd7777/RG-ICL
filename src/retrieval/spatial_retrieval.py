import numpy as np
import hashlib
from .global_retrieval import RetrievalResult


class SpatialRetriever:
    def __init__(self, similarity_metric: str = "cosine", exclude_query: bool = True,
                 exclude_test_set: bool = True):
        self.similarity_metric = similarity_metric
        self.exclude_query = exclude_query
        self.exclude_test_set = exclude_test_set
        self._index_spatial = None
        self._index_ids = None
        self._index_labels = None
        self._index_splits = None

    def build_index(self, ids: list, spatial_features: list, labels: list, splits: list):
        self._index_ids = np.array(ids)
        self._index_labels = np.array(labels)
        self._index_splits = np.array(splits)

        self._index_spatial = []
        for feat in spatial_features:
            norms = np.linalg.norm(feat, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._index_spatial.append(feat / norms)

    def _compute_spatial_similarity(self, query_spatial: np.ndarray, ref_spatial: np.ndarray) -> float:
        q_norms = np.linalg.norm(query_spatial, axis=1, keepdims=True)
        q_norms = np.maximum(q_norms, 1e-8)
        query_normalized = query_spatial / q_norms

        sim_matrix = query_normalized @ ref_spatial.T

        max_per_query_patch = sim_matrix.max(axis=1)
        score = float(max_per_query_patch.mean())
        return score

    def retrieve(self, query_id: str, query_spatial: np.ndarray, k: int = 6,
                 encoder_name: str = "", encoder_version: str = "",
                 preprocessing_hash: str = "") -> RetrievalResult:
        mask = np.ones(len(self._index_ids), dtype=bool)
        if self.exclude_query:
            mask &= self._index_ids != query_id
        if self.exclude_test_set:
            mask &= self._index_splits != "test"

        scores = np.full(len(self._index_ids), -np.inf)
        valid_indices = np.where(mask)[0]

        for idx in valid_indices:
            scores[idx] = self._compute_spatial_similarity(query_spatial, self._index_spatial[idx])

        top_indices = np.argsort(scores)[::-1][:k]

        neighbor_ids = self._index_ids[top_indices].tolist()
        neighbor_scores = scores[top_indices].tolist()
        neighbor_labels = self._index_labels[top_indices].tolist()

        emb_hash = hashlib.sha256(query_spatial.tobytes()).hexdigest()[:16]

        return RetrievalResult(
            query_id=query_id,
            query_embedding_hash=emb_hash,
            neighbor_ids=neighbor_ids,
            neighbor_scores=neighbor_scores,
            neighbor_labels=neighbor_labels,
            encoder_name=encoder_name,
            encoder_version=encoder_version,
            preprocessing_hash=preprocessing_hash,
            method="spatial",
        )
