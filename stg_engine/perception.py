"""STG Perception Layer — CNN feature extraction for visual grids.

Phase 12: Visual perception bridge between CNN and symbolic graph.

Numpy-only implementation. No torch dependency.
Fixed edge-detection filters + online Hebbian learning for pattern filters.

Architecture:
  64x64 grid (16 colors) -> one-hot (16,64,64) -> conv filters -> pool -> 128-dim vector

Usage:
  from stg_engine.perception import extract_features, grid_hash, PerceptionIndex

  features = extract_features(grid, build_fixed_filters())
  index = PerceptionIndex()
  index.add("frame_abc", features)
  similar = index.query(new_features, top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

FEATURE_DIM = 128
MAX_COLORS = 16
FIXED_FILTER_COUNT = 16
LEARNABLE_FILTER_COUNT = 8
LEARNABLE_FILTER_SIZE = 5


# ═══════════════════════════════════════════════════════════════════
# Grid Utilities
# ═══════════════════════════════════════════════════════════════════


def grid_hash(grid: List[List[int]]) -> str:
    """Hash grid to 12-char hex string for identity.

    Uses MD5 of raw bytes. Collision-safe for ~10^6 frames.
    """
    raw = bytes(cell for row in grid for cell in row)
    return hashlib.md5(raw).hexdigest()[:12]


def one_hot_grid(grid: List[List[int]], max_colors: int = MAX_COLORS) -> np.ndarray:
    """Convert grid to one-hot tensor (C, H, W).

    Args:
        grid: HxW grid with integer values 0 to max_colors-1
        max_colors: number of color channels

    Returns:
        (max_colors, H, W) float32 binary tensor
    """
    arr = np.asarray(grid, dtype=np.int32)
    h, w = arr.shape
    one_hot = np.zeros((max_colors, h, w), dtype=np.float32)
    for c in range(max_colors):
        one_hot[c] = (arr == c).astype(np.float32)
    return one_hot


# ═══════════════════════════════════════════════════════════════════
# Fixed Filters (edge detection, gradients, etc.)
# ═══════════════════════════════════════════════════════════════════


def build_fixed_filters() -> np.ndarray:
    """Return (16, 3, 3) array of fixed edge-detection kernels.

    Filters:
      0: Sobel horizontal       8: Top-left corner
      1: Sobel vertical         9: Top-right corner
      2: Diagonal (/)           10: Bottom-left corner
      3: Diagonal (\\)          11: Bottom-right corner
      4: Laplacian              12: Horizontal bar
      5: Cross                  13: Vertical bar
      6: Box blur               14: Center surround
      7: Sharpen                15: Identity (passthrough)
    """
    filters = np.zeros((FIXED_FILTER_COUNT, 3, 3), dtype=np.float32)

    # 0: Sobel horizontal
    filters[0] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # 1: Sobel vertical
    filters[1] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # 2: Diagonal (/)
    filters[2] = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # 3: Diagonal (\)
    filters[3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # 4: Laplacian
    filters[4] = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    # 5: Cross
    filters[5] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    # 6: Box blur (1/9 each)
    filters[6] = np.ones((3, 3), dtype=np.float32) / 9.0
    # 7: Sharpen
    filters[7] = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    # 8-11: Corner detectors
    filters[8] = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]   # top-left
    filters[9] = [[0, 0, 1], [0, -1, 0], [0, 0, 0]]   # top-right
    filters[10] = [[0, 0, 0], [0, -1, 0], [1, 0, 0]]  # bottom-left
    filters[11] = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]  # bottom-right
    # 12: Horizontal bar
    filters[12] = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    # 13: Vertical bar
    filters[13] = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    # 14: Center surround
    filters[14] = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    # 15: Identity (passthrough)
    filters[15] = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    return filters


# ═══════════════════════════════════════════════════════════════════
# Learnable Filters
# ═══════════════════════════════════════════════════════════════════


def init_learnable_filters(
    count: int = LEARNABLE_FILTER_COUNT,
    size: int = LEARNABLE_FILTER_SIZE,
    seed: int = 42,
) -> np.ndarray:
    """Initialize learnable filters with random Gabor-like patterns.

    Returns: (count, size, size) float32, L2-normalized per filter
    """
    rng = np.random.RandomState(seed)
    filters = rng.randn(count, size, size).astype(np.float32)
    # L2 normalize each filter
    for i in range(count):
        norm = np.linalg.norm(filters[i])
        if norm > 0:
            filters[i] /= norm
    return filters


# ═══════════════════════════════════════════════════════════════════
# Convolution (numpy only)
# ═══════════════════════════════════════════════════════════════════


def conv2d(input_map: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution via manual sliding window. Same-size output (zero-padded).

    Args:
        input_map: (H, W) float32
        kernel: (kH, kW) float32

    Returns:
        (H, W) float32
    """
    h, w = input_map.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Zero-pad input
    padded = np.pad(input_map, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # Use stride_tricks for efficient sliding window
    ph, pw = padded.shape
    out_h, out_w = ph - kh + 1, pw - kw + 1
    strides = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(out_h, out_w, kh, kw),
        strides=(strides[0], strides[1], strides[0], strides[1]),
    )

    # Element-wise multiply and sum
    output = np.einsum("ijkl,kl->ij", windows, kernel)
    return output[:h, :w]


def apply_filters(
    one_hot: np.ndarray,
    filters: np.ndarray,
) -> np.ndarray:
    """Apply bank of filters across all input channels, sum across channels.

    For each filter f:
        activation[f] = ReLU(sum_c conv2d(one_hot[c], filters[f]))

    Args:
        one_hot: (C, H, W) one-hot encoded input
        filters: (F, kH, kW) filter bank

    Returns:
        (F, H, W) float32 activation maps (ReLU applied)
    """
    c, h, w = one_hot.shape
    f_count = filters.shape[0]
    activations = np.zeros((f_count, h, w), dtype=np.float32)

    for f_idx in range(f_count):
        for c_idx in range(c):
            # Skip empty channels for efficiency
            if one_hot[c_idx].max() == 0:
                continue
            activations[f_idx] += conv2d(one_hot[c_idx], filters[f_idx])

    # ReLU
    np.maximum(activations, 0, out=activations)
    return activations


# ═══════════════════════════════════════════════════════════════════
# Feature Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════


def _color_histogram(grid: List[List[int]], max_colors: int = MAX_COLORS) -> np.ndarray:
    """Normalized color histogram (max_colors,) float32."""
    arr = np.asarray(grid, dtype=np.int32).ravel()
    hist = np.bincount(arr, minlength=max_colors).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist[:max_colors]


def _spatial_stats(
    grid: List[List[int]],
    fixed_activations: np.ndarray,
    learnable_activations: Optional[np.ndarray],
) -> np.ndarray:
    """Spatial statistics: 4x4 grid features + global features = 88 dim.

    4x4 grid x 5 features = 80:
        per cell: dominant_color, density, edge_strength, pattern_activation, entropy
    Global features = 8:
        nonzero_count, h_symmetry, v_symmetry, d1_symmetry, bbox_x, bbox_y, bbox_w, bbox_h
    """
    arr = np.asarray(grid, dtype=np.int32)
    h, w = arr.shape

    # 4x4 spatial grid features (80 dim)
    cell_h, cell_w = h // 4, w // 4
    grid_features = np.zeros(80, dtype=np.float32)

    for gy in range(4):
        for gx in range(4):
            cell = arr[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            idx = (gy * 4 + gx) * 5

            # Dominant color (normalized to 0-1)
            hist = np.bincount(cell.ravel(), minlength=MAX_COLORS)
            grid_features[idx] = float(np.argmax(hist)) / MAX_COLORS

            # Density (fraction of non-zero pixels)
            grid_features[idx + 1] = float(np.count_nonzero(cell)) / cell.size

            # Edge strength (mean of fixed filter activations in this cell)
            if fixed_activations is not None:
                cell_act = fixed_activations[
                    :, gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w
                ]
                grid_features[idx + 2] = float(cell_act.mean())

            # Pattern activation (mean of learnable filter activations)
            if learnable_activations is not None:
                cell_pat = learnable_activations[
                    :, gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w
                ]
                grid_features[idx + 3] = float(cell_pat.mean())

            # Entropy (color diversity)
            probs = hist.astype(np.float32)
            total = probs.sum()
            if total > 0:
                probs = probs / total
                probs = probs[probs > 0]
                grid_features[idx + 4] = float(-np.sum(probs * np.log2(probs)))

    # Global features (8 dim)
    global_features = np.zeros(8, dtype=np.float32)

    # Non-zero pixel count (normalized)
    global_features[0] = float(np.count_nonzero(arr)) / arr.size

    # Symmetry scores
    # Horizontal symmetry
    flipped_h = np.fliplr(arr)
    global_features[1] = float(np.mean(arr == flipped_h))
    # Vertical symmetry
    flipped_v = np.flipud(arr)
    global_features[2] = float(np.mean(arr == flipped_v))
    # Diagonal symmetry (transpose)
    if h == w:
        global_features[3] = float(np.mean(arr == arr.T))

    # Bounding box of non-background (assuming most common color is background)
    bg_hist = np.bincount(arr.ravel(), minlength=MAX_COLORS)
    bg_color = int(np.argmax(bg_hist))
    non_bg = np.argwhere(arr != bg_color)
    if len(non_bg) > 0:
        min_y, min_x = non_bg.min(axis=0)
        max_y, max_x = non_bg.max(axis=0)
        global_features[4] = float(min_x) / w
        global_features[5] = float(min_y) / h
        global_features[6] = float(max_x - min_x + 1) / w
        global_features[7] = float(max_y - min_y + 1) / h

    return np.concatenate([grid_features, global_features])


def extract_features(
    grid: List[List[int]],
    fixed_filters: np.ndarray,
    learnable_filters: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full pipeline: grid -> 128-dim L2-normalized feature vector.

    Components (128 total):
        [0:16]   - Fixed filter global average activations (16 dim)
        [16:24]  - Learnable filter global average activations (8 dim)
        [24:40]  - Normalized color histogram (16 dim)
        [40:128] - Spatial statistics (88 dim)

    Args:
        grid: HxW grid with integer color values
        fixed_filters: (16, 3, 3) fixed edge-detection filters
        learnable_filters: optional (8, 5, 5) learnable pattern filters

    Returns:
        (128,) float32, L2-normalized
    """
    one_hot = one_hot_grid(grid)

    # Fixed filter activations
    fixed_act = apply_filters(one_hot, fixed_filters)
    edge_features = np.array(
        [fixed_act[i].mean() for i in range(fixed_act.shape[0])],
        dtype=np.float32,
    )

    # Learnable filter activations
    if learnable_filters is not None:
        learn_act = apply_filters(one_hot, learnable_filters)
        pattern_features = np.array(
            [learn_act[i].mean() for i in range(learn_act.shape[0])],
            dtype=np.float32,
        )
    else:
        learn_act = None
        pattern_features = np.zeros(LEARNABLE_FILTER_COUNT, dtype=np.float32)

    # Color histogram
    color_hist = _color_histogram(grid)

    # Spatial statistics
    spatial = _spatial_stats(grid, fixed_act, learn_act)

    # Concatenate
    feature_vector = np.concatenate([edge_features, pattern_features, color_hist, spatial])

    # Ensure correct dimension
    assert feature_vector.shape == (FEATURE_DIM,), (
        f"Feature vector dim mismatch: {feature_vector.shape} != ({FEATURE_DIM},)"
    )

    # L2 normalize
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector /= norm

    return feature_vector


# ═══════════════════════════════════════════════════════════════════
# Similarity Search
# ═══════════════════════════════════════════════════════════════════


class PerceptionIndex:
    """In-memory cosine similarity index for perception feature vectors.

    Stores {frame_hash: feature_vector} pairs.
    Query returns top-k most similar frames by cosine similarity.
    """

    def __init__(self) -> None:
        self._keys: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # (N, D) row-normalized

    def build(self, features: Dict[str, np.ndarray]) -> None:
        """Build index from a dict of {hash: vector}."""
        self._keys = list(features.keys())
        if self._keys:
            self._matrix = np.stack([features[k] for k in self._keys])
        else:
            self._matrix = None

    def add(self, frame_hash: str, vector: np.ndarray) -> None:
        """Add a single vector to the index."""
        if frame_hash in self._keys:
            # Update existing
            idx = self._keys.index(frame_hash)
            self._matrix[idx] = vector
            return

        self._keys.append(frame_hash)
        row = vector.reshape(1, -1)
        if self._matrix is None:
            self._matrix = row
        else:
            self._matrix = np.vstack([self._matrix, row])

    def remove(self, frame_hash: str) -> bool:
        """Remove a vector from the index."""
        if frame_hash not in self._keys:
            return False
        idx = self._keys.index(frame_hash)
        self._keys.pop(idx)
        if self._matrix is not None:
            self._matrix = np.delete(self._matrix, idx, axis=0)
            if self._matrix.shape[0] == 0:
                self._matrix = None
        return True

    def query(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find top-k most similar vectors by cosine similarity.

        Args:
            query_vector: (D,) L2-normalized query
            top_k: number of results

        Returns:
            List of (frame_hash, similarity) sorted descending
        """
        if self._matrix is None or len(self._keys) == 0:
            return []

        # Cosine similarity (vectors are L2-normalized)
        sims = self._matrix @ query_vector
        top_k = min(top_k, len(self._keys))
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(self._keys[i], float(sims[i])) for i in top_indices]

    @property
    def size(self) -> int:
        return len(self._keys)


# ═══════════════════════════════════════════════════════════════════
# Hebbian Online Learning
# ═══════════════════════════════════════════════════════════════════


def update_filters_hebbian(
    filters: np.ndarray,
    one_hot_input: np.ndarray,
    activations: np.ndarray,
    reward: float,
    learning_rate: float = 0.01,
) -> np.ndarray:
    """Hebbian update: reinforce filters correlated with positive outcomes.

    Simplified Oja's rule:
        delta_f = lr * reward * mean(activation_f * input_patches_under_f)
        f_new = normalize(f + delta_f)

    Args:
        filters: (F, kH, kW) current learnable filters
        one_hot_input: (C, H, W) one-hot encoded input
        activations: (F, H, W) post-conv activations for learnable filters
        reward: +1.0 for level-up, -0.1 for stuck, 0 for neutral
        learning_rate: step size

    Returns:
        Updated (F, kH, kW) filters, L2-normalized
    """
    if abs(reward) < 1e-8:
        return filters

    f_count, kh, kw = filters.shape
    updated = filters.copy()

    for f_idx in range(f_count):
        # Mean activation as weight
        act_mean = activations[f_idx].mean()
        if act_mean < 1e-8:
            continue

        # Compute correlation between filter activation and input
        # Use mean of input across channels as proxy
        input_mean = one_hot_input.mean(axis=0)  # (H, W)

        # Crop center patch matching filter size
        h, w = input_mean.shape
        cy, cx = h // 2, w // 2
        ph, pw = kh // 2, kw // 2
        patch = input_mean[cy - ph : cy + ph + 1, cx - pw : cx + pw + 1]

        if patch.shape == (kh, kw):
            delta = learning_rate * reward * act_mean * patch
            updated[f_idx] += delta

    # L2 normalize each filter
    for i in range(f_count):
        norm = np.linalg.norm(updated[i])
        if norm > 0:
            updated[i] /= norm

    return updated


# ═══════════════════════════════════════════════════════════════════
# Engine Integration Helpers
# ═══════════════════════════════════════════════════════════════════


def perceive_frame(
    engine: "STGEngine",
    grid: List[List[int]],
    game_id: Optional[str] = None,
    step_number: int = 0,
    level: int = 0,
) -> Tuple[str, np.ndarray]:
    """Perceive a grid frame: extract features, create STG node, index vector.

    1. Compute grid_hash
    2. If hash already in index -> return cached
    3. Extract features
    4. Create STGNode: Visual:frame_{hash}, anchor_type=Perception
    5. Add to PerceptionIndex

    Returns: (frame_hash, 128-dim feature_vector)
    """
    fhash = grid_hash(grid)
    node_name = f"Visual:frame_{fhash}"

    # Lazy init perception state on engine
    if not hasattr(engine, "_fixed_filters") or engine._fixed_filters is None:
        engine._fixed_filters = build_fixed_filters()
    if not hasattr(engine, "_perception_filters"):
        engine._perception_filters = None
    if not hasattr(engine, "_perception_index"):
        engine._perception_index = PerceptionIndex()

    # Check if already indexed
    if engine._perception_index is not None:
        for key, _ in engine._perception_index.query(
            np.zeros(FEATURE_DIM), top_k=engine._perception_index.size
        ):
            if key == fhash:
                # Already indexed, extract features for return
                features = extract_features(
                    grid, engine._fixed_filters, engine._perception_filters
                )
                return fhash, features

    # Extract features
    features = extract_features(
        grid, engine._fixed_filters, engine._perception_filters
    )

    # Create STG node (if not exists)
    h = len(grid)
    w = len(grid[0]) if grid else 0
    color_hist = _color_histogram(grid)
    n_colors = int(np.count_nonzero(color_hist))

    if node_name not in engine._graph:
        engine.add_node(
            name=node_name,
            namespace="Visual",
            anchor_type="Perception",
            game_id=game_id or "",
            step_number=step_number,
            level=level,
            grid_size=f"{w}x{h}",
            n_colors=n_colors,
        )

    # Add to perception index
    engine._perception_index.add(fhash, features)

    return fhash, features


def record_transition(
    engine: "STGEngine",
    from_hash: str,
    to_hash: str,
    action_id: int,
    reward: float = 0.0,
    level_delta: int = 0,
) -> None:
    """Record state transition as STG edge between perception nodes.

    Creates edge:
        [Visual:frame_{from}] -> [Visual:frame_{to}]
        ::mod(confidence=1.0, rule="causal", action={action_id},
              reward={reward}, level_delta={level_delta},
              edge_class="temporal", delay_k=1)
    """
    from_node = f"Visual:frame_{from_hash}"
    to_node = f"Visual:frame_{to_hash}"

    salience = 1.0 if level_delta > 0 else 0.5

    engine.add_edge(
        source=from_node,
        target=to_node,
        confidence=1.0,
        salience=salience,
        rule="causal",
        edge_class="temporal",
        delay_k=1,
        action=action_id,
        reward=reward,
        level_delta=level_delta,
    )


def find_similar_states(
    engine: "STGEngine",
    grid: List[List[int]],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Find visually similar past states.

    Returns: List of (node_name, similarity) sorted descending
    """
    if not hasattr(engine, "_perception_index") or engine._perception_index is None:
        return []
    if not hasattr(engine, "_fixed_filters") or engine._fixed_filters is None:
        engine._fixed_filters = build_fixed_filters()

    features = extract_features(
        grid,
        engine._fixed_filters,
        getattr(engine, "_perception_filters", None),
    )
    results = engine._perception_index.query(features, top_k=top_k)
    return [(f"Visual:frame_{fhash}", sim) for fhash, sim in results]


# ═══════════════════════════════════════════════════════════════════
# Autonomous Perception Loop
# ═══════════════════════════════════════════════════════════════════


def select_action_from_memory(
    engine: "STGEngine",
    current_hash: str,
    available_actions: List[int],
    explore_rate: float = 0.2,
) -> Tuple[int, str]:
    """STG-driven action selection: recall what worked in similar states.

    Looks at edges from similar frames, weighted by:
    - similarity to current frame (from PerceptionIndex)
    - historical reward of the action
    - salience (Hebbian-reinforced)

    Falls back to random exploration if no useful memory.

    Args:
        engine: STGEngine with perception state
        current_hash: hash of current frame
        available_actions: list of valid action IDs
        explore_rate: probability of random exploration even with memory

    Returns:
        (action_id, reason) where reason explains the choice
    """
    import random

    if not available_actions:
        return 0, "no_actions_available"

    # Random exploration with probability explore_rate
    if random.random() < explore_rate:
        action = random.choice(available_actions)
        return action, "exploration"

    # Find similar past frames
    if not hasattr(engine, "_perception_index") or engine._perception_index is None:
        action = random.choice(available_actions)
        return action, "no_index"

    current_node = f"Visual:frame_{current_hash}"

    # Collect action scores from edges of current and similar frames
    action_scores: Dict[int, float] = {}
    action_counts: Dict[int, int] = {}

    # Check edges from current frame and nearby frames
    nodes_to_check = [current_node]

    # Also check similar frames' outgoing edges
    if engine._perception_index.size > 0:
        # Get current features to find similar
        features_dict = {}
        for key in engine._perception_index._keys:
            if key == current_hash:
                continue
            idx = engine._perception_index._keys.index(key)
            if engine._perception_index._matrix is not None:
                features_dict[key] = engine._perception_index._matrix[idx]

        if features_dict and current_hash in engine._perception_index._keys:
            idx = engine._perception_index._keys.index(current_hash)
            query_vec = engine._perception_index._matrix[idx]
            similar = engine._perception_index.query(query_vec, top_k=5)
            for fhash, sim in similar:
                if fhash != current_hash:
                    nodes_to_check.append(f"Visual:frame_{fhash}")

    # Scan outgoing edges for action/reward data
    for node_name in nodes_to_check:
        if node_name not in engine._nodes:
            continue
        if not engine._graph.has_node(node_name):
            continue
        for succ in engine._graph.successors(node_name):
            edge = engine._edges_lookup.get((node_name, succ))
            if edge is None:
                continue
            action_id = edge.modifiers.get("action")
            if action_id is None:
                continue
            action_id = int(action_id)
            if action_id not in available_actions:
                continue

            reward = float(edge.modifiers.get("reward", 0.0))
            sal = edge.salience

            # Score = reward * salience (reinforced paths score higher)
            score = reward * sal
            action_scores[action_id] = action_scores.get(action_id, 0.0) + score
            action_counts[action_id] = action_counts.get(action_id, 0) + 1

    # If we have scored actions, pick the best
    if action_scores:
        # Normalize by count
        for a in action_scores:
            if action_counts[a] > 0:
                action_scores[a] /= action_counts[a]

        best_action = max(action_scores, key=lambda a: action_scores[a])
        best_score = action_scores[best_action]

        if best_score > 0:
            return best_action, f"memory(score={best_score:.3f},n={action_counts[best_action]})"

        # All scores <= 0: avoid known-bad actions, try untried ones
        tried = set(action_scores.keys())
        untried = [a for a in available_actions if a not in tried]
        if untried:
            action = random.choice(untried)
            return action, "untried_action"

    # No memory at all — random exploration
    action = random.choice(available_actions)
    return action, "no_memory"


def perception_step(
    engine: "STGEngine",
    grid: List[List[int]],
    available_actions: List[int],
    prev_hash: Optional[str] = None,
    prev_action: Optional[int] = None,
    prev_level: int = 0,
    current_level: int = 0,
    game_id: Optional[str] = None,
    step_number: int = 0,
    explore_rate: float = 0.2,
    pixels_changed: int = -1,
) -> Tuple[int, str, str]:
    """One step of the autonomous STG perception loop.

    This is the core cycle:
        STG perceives → recalls → selects action → records transition → learns

    Call this in a game loop. STG drives the decision, no LLM needed.

    Args:
        engine: STGEngine with perception state
        grid: current game frame (HxW grid)
        available_actions: valid action IDs for this step
        prev_hash: hash of previous frame (None on first step)
        prev_action: action taken last step (None on first step)
        prev_level: level before this step
        current_level: level after last action
        game_id: game identifier
        step_number: current step number
        explore_rate: random exploration probability
        pixels_changed: number of pixels changed from previous frame (-1 = unknown)

    Returns:
        (action_id, frame_hash, reason)
        - action_id: the action STG chose
        - frame_hash: hash of current frame (pass as prev_hash next step)
        - reason: why this action was chosen
    """
    # 1. CNN sees — extract features, create/update node
    fhash, features = perceive_frame(
        engine, grid, game_id=game_id,
        step_number=step_number, level=current_level,
    )

    # 2. Record previous transition (if not first step)
    if prev_hash is not None and prev_action is not None:
        level_delta = current_level - prev_level
        reward = 0.0
        if level_delta > 0:
            reward = 1.0  # level up
        elif fhash == prev_hash:
            reward = -0.2  # no change (stuck)
        elif pixels_changed >= 0 and pixels_changed < 10:
            reward = -0.1  # barely changed (noise/wall bounce)
        else:
            reward = 0.1  # meaningful new state

        record_transition(
            engine, prev_hash, fhash,
            action_id=prev_action,
            reward=reward,
            level_delta=level_delta,
        )

        # 3. Hebbian filter learning on reward
        if abs(reward) > 0.05 and hasattr(engine, "_perception_filters"):
            if engine._perception_filters is not None:
                one_hot = one_hot_grid(grid)
                learn_act = apply_filters(one_hot, engine._perception_filters)
                engine._perception_filters = update_filters_hebbian(
                    engine._perception_filters,
                    one_hot, learn_act,
                    reward=reward,
                )

    # 4. STG recalls and selects action
    action_id, reason = select_action_from_memory(
        engine, fhash, available_actions, explore_rate=explore_rate,
    )

    return action_id, fhash, reason


# ═══════════════════════════════════════════════════════════════════
# LLM Teaching Interface (STG calls LLM when it needs help)
# ═══════════════════════════════════════════════════════════════════


def _grid_to_text(grid: List[List[int]], max_rows: int = 64) -> str:
    """Convert grid to compact text representation for LLM.

    Uses hex digits (0-f) for colors, one row per line.
    """
    lines = []
    for y, row in enumerate(grid[:max_rows]):
        lines.append("".join(f"{c:x}" for c in row))
    return "\n".join(lines)


def _grid_to_ascii_art(grid: List[List[int]], scale: int = 1) -> str:
    """Convert grid to ASCII art with shape outlines.

    Groups same-color regions and marks boundaries.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    art = []
    symbols = " .:-=+*#@"
    for y in range(0, h, scale):
        row = ""
        for x in range(0, w, scale):
            c = grid[y][x]
            # Check if this is a boundary (neighbor has different color)
            is_edge = False
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] != c:
                    is_edge = True
                    break
            if is_edge:
                row += f"{c:x}"  # show color at edges
            else:
                row += " "  # interior is blank
        art.append(row)
    return "\n".join(art)


def needs_llm_help(
    engine: "STGEngine",
    grid: List[List[int]],
    similarity_threshold: float = 0.95,
) -> bool:
    """Determine if STG needs LLM help to understand this frame.

    Returns True if:
    - No similar frames in memory (completely new visual pattern)
    - Similar frames exist but have no semantic labels (shape, rule, etc.)
    """
    if not hasattr(engine, "_perception_index") or engine._perception_index is None:
        return True
    if engine._perception_index.size == 0:
        return True

    if not hasattr(engine, "_fixed_filters") or engine._fixed_filters is None:
        engine._fixed_filters = build_fixed_filters()

    features = extract_features(
        grid, engine._fixed_filters,
        getattr(engine, "_perception_filters", None),
    )
    # Get current frame hash to exclude self-match
    current_hash = grid_hash(grid)

    results = engine._perception_index.query(features, top_k=5)
    if not results:
        return True

    # Check all similar frames (not just top-1), skip self
    for fhash, sim in results:
        if fhash == current_hash:
            continue  # skip self
        if sim < similarity_threshold:
            break  # remaining results are even less similar

        # Check if this similar frame has semantic labels (non-Visual edges)
        node_name = f"Visual:frame_{fhash}"
        if node_name in engine._nodes and engine._graph.has_node(node_name):
            for succ in engine._graph.successors(node_name):
                if not succ.startswith("Visual:"):
                    return False  # has semantic labels, no help needed
            for pred in engine._graph.predecessors(node_name):
                if not pred.startswith("Visual:"):
                    return False

    return True  # no semantic labels found in similar frames


def build_teaching_prompt(
    grid: List[List[int]],
    game_id: Optional[str] = None,
    context: Optional[str] = None,
    node_name: Optional[str] = None,
) -> str:
    """Build a prompt for LLM to analyze a game frame in STL format.

    Returns a text prompt that describes the grid and asks for
    semantic labels (shapes, objects, spatial relationships) as STL.
    """
    h, w = len(grid), len(grid[0]) if grid else 0

    if node_name is None:
        node_name = f"Visual:frame_{grid_hash(grid)}"

    # Color histogram
    from collections import Counter
    color_counts = Counter(c for row in grid for c in row)
    total = h * w
    colors_used = sorted(color_counts.keys())

    # Find non-background objects
    bg_color = color_counts.most_common(1)[0][0]
    objects = {}
    for c in colors_used:
        if c == bg_color:
            continue
        pixels = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == c]
        if pixels:
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            objects[c] = {
                "count": len(pixels),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "pct": len(pixels) / total * 100,
            }

    prompt = f"""Analyze this {w}x{h} game grid. Each number (hex) is a color.

Grid (hex digits, each = 1 pixel):
{_grid_to_text(grid)}

Background: color {bg_color:x} ({color_counts[bg_color]} pixels, {color_counts[bg_color]/total*100:.0f}%)

Objects found:
"""
    for c, info in objects.items():
        bx, by, bx2, by2 = info["bbox"]
        prompt += f"- Color {c:x}: {info['count']}px ({info['pct']:.1f}%), bbox=({bx},{by})-({bx2},{by2})\n"

    if context:
        prompt += f"\nContext: {context}\n"

    prompt += f"""
Describe what you see using STL (Semantic Tension Language) format.
Source anchor must be [{node_name}] where {node_name} is this frame.

STL syntax: [Source] -> [Target] ::mod(key=value, ...)
Required modifiers: confidence (0-1), rule ("empirical"/"causal"/"definitional")
Useful modifiers: color, position, description, size

Example output:
[{node_name}] -> [Shape:Square] ::mod(confidence=0.90, rule="empirical", color="9", position="top-left", size="8x8")
[{node_name}] -> [Object:Movable_Block] ::mod(confidence=0.85, rule="empirical", color="c", position="center")
[{node_name}] -> [Rule:Move_Block_With_Arrows] ::mod(confidence=0.70, rule="empirical", description="arrow keys move the colored block")
[Shape:Square] -> [Object:Movable_Block] ::mod(confidence=0.80, rule="causal", description="block is inside the square")

Output ONLY STL statements, one per line. No other text.
"""
    return prompt


async def teach_from_llm(
    engine: "STGEngine",
    grid: List[List[int]],
    adapter: Any,  # LLMAdapter
    game_id: Optional[str] = None,
    context: Optional[str] = None,
) -> List[str]:
    """Call LLM to analyze a frame and store labels in STG.

    This is the "mom says 'cat'" moment — LLM teaches STG
    what it sees, STG remembers, never asks again for same pattern.

    Args:
        engine: STGEngine with perception state
        grid: game frame
        adapter: SKC LLMAdapter instance
        game_id: optional game identifier
        context: optional context string

    Returns:
        List of STL statements ingested into STG
    """
    # Ensure frame is perceived
    fhash, _ = perceive_frame(engine, grid, game_id=game_id)

    # Build prompt and call LLM
    node_name = f"Visual:frame_{fhash}"
    prompt = build_teaching_prompt(grid, game_id, context, node_name=node_name)

    # Import SKC types
    try:
        from skc.types.llm import Message
    except ImportError:
        raise NotImplementedError("This LLM-dependent feature requires the SKC package. Use stg-engine for the core algorithms; SKC for LLM integration.")

    messages = [Message(role="user", content=prompt)]
    response = await adapter.generate_response(
        messages,
        system_prompt=(
            "You are a visual pattern analyzer for abstract grid games. "
            "Output ONLY valid STL statements. "
            "STL syntax: [Source] -> [Target] ::mod(key=value, ...) "
            "Do not output any other text. "
            "No thinking. Just output."
        ),
    )

    # Use stl_parser's validate_llm_output to clean and repair LLM output
    from stl_parser import validate_llm_output

    result = validate_llm_output(response.content)

    # Ingest repaired STL statements into STG
    # cleaned_text contains all repaired statements as valid STL
    stl_statements = []
    if result.statements:
        for line in result.cleaned_text.strip().split("\n"):
            line = line.strip()
            if not line or "->" not in line:
                continue
            stl_statements.append(line)
            try:
                engine.ingest_stl(line)
            except Exception:
                pass  # skip statements that still fail after repair

    return stl_statements
