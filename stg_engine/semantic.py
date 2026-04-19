"""STG Semantic Query Layer (Phase 7G).

Adds multilingual semantic search to STG Engine via embedding vectors.
Two-phase search: Flash (vector similarity) + Unfold (graph propagation).

Dependencies:
    - sentence-transformers (optional, for real embeddings)
    - numpy (required, for vector operations)

If sentence-transformers is not installed, all existing STG functionality
works normally. Only search() and embed commands require it.
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Model name constant — change here to switch models globally
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


class EmbeddingBuilder:
    """Composes rich embedding text from node context.

    A node name alone ('Wuko_Paradigm_Shift') carries too little semantic
    information for accurate embedding. This class composes multi-layer
    text from the node's name, namespace, metadata, and 1-hop neighbors.
    """

    @staticmethod
    def unpack_name(name: str) -> str:
        """Convert node names to natural text.

        'Wuko_Paradigm_Shift' → 'Wuko Paradigm Shift'
        'MemoryManager'       → 'Memory Manager'
        'Physics:Energy'      → 'Physics Energy'

        Args:
            name: Node name in any convention

        Returns:
            Space-separated natural text
        """
        # Replace namespace separator
        text = name.replace(":", " ")
        # Replace underscores
        text = text.replace("_", " ")
        # Split PascalCase: 'MemoryManager' → 'Memory Manager'
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
        # Clean up extra spaces
        return " ".join(text.split())

    @staticmethod
    def build_embed_text(
        node_name: str,
        namespace: Optional[str],
        metadata: Dict[str, Any],
        neighbor_names: List[str],
        edge_metadata: List[Dict[str, Any]],
    ) -> str:
        """Compose embedding text from node + neighborhood.

        Layers:
          1. Node name (unpacked to natural words)
          2. Namespace
          3. Node metadata (type, description, tags)
          4. 1-hop neighbor names + edge semantic keys

        Args:
            node_name: The node's name
            namespace: Optional namespace
            metadata: Node metadata dict
            neighbor_names: Names of 1-hop neighbors
            edge_metadata: Metadata dicts from edges to/from neighbors

        Returns:
            Space-joined text suitable for embedding model input
        """
        parts = []

        # Layer 1: Node name
        parts.append(EmbeddingBuilder.unpack_name(node_name))

        # Layer 2: Namespace
        if namespace:
            parts.append(namespace)

        # Layer 3: Metadata
        for key in ("type", "description", "tags", "anchor_type"):
            if key in metadata:
                val = metadata[key]
                if isinstance(val, list):
                    parts.extend(str(v) for v in val)
                elif val:
                    parts.append(str(val))

        # Layer 4: Neighbor context (cap at 10 to avoid overly long text)
        for name in neighbor_names[:10]:
            parts.append(EmbeddingBuilder.unpack_name(name))

        # Layer 5: Edge semantic keys
        for emeta in edge_metadata[:10]:
            for key in ("rule", "cause", "effect"):
                if key in emeta and emeta[key]:
                    parts.append(str(emeta[key]))

        return " ".join(parts)

    def build_all(self, engine) -> Dict[str, str]:
        """Build embed text for all nodes in the engine.

        Args:
            engine: STGEngine instance

        Returns:
            Dict mapping node_name → embed_text
        """
        result = {}
        for name, node in engine._nodes.items():
            # Gather neighbor info
            neighbor_names = []
            edge_metas = []
            for succ in engine._graph.successors(name):
                neighbor_names.append(succ)
                edge = engine._edges_lookup.get((name, succ))
                if edge:
                    edge_metas.append(edge.modifiers)
            for pred in engine._graph.predecessors(name):
                if pred not in neighbor_names:
                    neighbor_names.append(pred)
                    edge = engine._edges_lookup.get((pred, name))
                    if edge:
                        edge_metas.append(edge.modifiers)

            result[name] = self.build_embed_text(
                node_name=name,
                namespace=node.namespace,
                metadata=node.metadata,
                neighbor_names=neighbor_names,
                edge_metadata=edge_metas,
            )
        return result


class VectorIndex:
    """In-memory vector index for cosine similarity search.

    Uses normalized vectors + dot product (equivalent to cosine similarity).
    For 4K nodes with 384-dim vectors, numpy is sufficient — no FAISS needed.
    """

    def __init__(self):
        self.names: List[str] = []
        self.matrix: Optional[np.ndarray] = None  # (N, dim) float32, L2-normalized

    def build(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Build index from pre-computed embeddings.

        Args:
            embeddings: Dict mapping node_name → vector (normalized)
        """
        if not embeddings:
            self.names = []
            self.matrix = None
            return

        self.names = list(embeddings.keys())
        vectors = [embeddings[name] for name in self.names]
        self.matrix = np.vstack(vectors).astype(np.float32)

    def query(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top-K nodes by cosine similarity.

        Args:
            query_vector: Normalized query vector (dim,)
            top_k: Number of results

        Returns:
            List of (node_name, similarity_score) sorted descending
        """
        if self.matrix is None or len(self.names) == 0:
            return []

        query_vector = query_vector.astype(np.float32).flatten()
        scores = self.matrix @ query_vector  # (N,)

        k = min(top_k, len(self.names))
        if k <= 0:
            return []

        # Get top-K indices
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.names[i], float(scores[i])) for i in top_indices]

    def add(self, name: str, vector: np.ndarray) -> None:
        """Add a single node to the index (incremental update).

        Args:
            name: Node name
            vector: Normalized vector (dim,)
        """
        vector = vector.astype(np.float32).reshape(1, -1)

        if self.matrix is None:
            self.names = [name]
            self.matrix = vector
        else:
            # Remove if already exists (update case)
            if name in self.names:
                self.remove(name)
            self.names.append(name)
            self.matrix = np.vstack([self.matrix, vector])

    def remove(self, name: str) -> bool:
        """Remove a node from the index.

        Args:
            name: Node name to remove

        Returns:
            True if node was found and removed
        """
        if name not in self.names:
            return False

        idx = self.names.index(name)
        self.names.pop(idx)
        if self.matrix is not None:
            self.matrix = np.delete(self.matrix, idx, axis=0)
            if len(self.names) == 0:
                self.matrix = None
        return True

    @property
    def size(self) -> int:
        """Number of indexed nodes."""
        return len(self.names)


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load a sentence-transformers model.

    Args:
        model_name: HuggingFace model name

    Returns:
        SentenceTransformer model instance

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for semantic search. "
            "Install with: pip install sentence-transformers"
        )
    return SentenceTransformer(model_name)
