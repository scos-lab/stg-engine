"""Tests for STG Perception Layer (Phase 12)."""

import numpy as np
import pytest

from stg_engine.perception import (
    FEATURE_DIM,
    FIXED_FILTER_COUNT,
    LEARNABLE_FILTER_COUNT,
    MAX_COLORS,
    PerceptionIndex,
    apply_filters,
    build_fixed_filters,
    conv2d,
    extract_features,
    grid_hash,
    init_learnable_filters,
    one_hot_grid,
    update_filters_hebbian,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def fixed_filters():
    return build_fixed_filters()


@pytest.fixture
def learnable_filters():
    return init_learnable_filters()


@pytest.fixture
def simple_grid():
    """8x8 grid with a simple pattern for testing."""
    grid = [[0] * 8 for _ in range(8)]
    # Draw a square in the center
    for y in range(2, 6):
        for x in range(2, 6):
            grid[y][x] = 5
    return grid


@pytest.fixture
def arc_grid():
    """64x64 grid mimicking ARC-AGI-3 game state."""
    grid = [[4] * 64 for _ in range(64)]
    # Floor area
    for y in range(25, 50):
        for x in range(14, 54):
            grid[y][x] = 3
    # Movable block (orange + blue)
    for y in range(45, 50):
        for x in range(34, 39):
            grid[y][x] = 12 if y < 47 else 9
    return grid


@pytest.fixture
def arc_grid_shifted(arc_grid):
    """Same structure as arc_grid but block shifted right by 5."""
    grid = [row[:] for row in arc_grid]
    # Clear old block position
    for y in range(45, 50):
        for x in range(34, 39):
            grid[y][x] = 3
    # New position (shifted right)
    for y in range(45, 50):
        for x in range(39, 44):
            grid[y][x] = 12 if y < 47 else 9
    return grid


# ═══════════════════════════════════════════════════════════════════
# TestOneHotEncoding
# ═══════════════════════════════════════════════════════════════════


class TestOneHotEncoding:
    def test_shape(self, simple_grid):
        oh = one_hot_grid(simple_grid)
        assert oh.shape == (MAX_COLORS, 8, 8)

    def test_dtype(self, simple_grid):
        oh = one_hot_grid(simple_grid)
        assert oh.dtype == np.float32

    def test_binary_values(self, simple_grid):
        oh = one_hot_grid(simple_grid)
        assert set(np.unique(oh)) == {0.0, 1.0}

    def test_correct_encoding(self):
        grid = [[0, 1], [2, 3]]
        oh = one_hot_grid(grid)
        assert oh[0, 0, 0] == 1.0  # color 0 at (0,0)
        assert oh[1, 0, 1] == 1.0  # color 1 at (0,1)
        assert oh[2, 1, 0] == 1.0  # color 2 at (1,0)
        assert oh[3, 1, 1] == 1.0  # color 3 at (1,1)
        # Non-matching channels should be 0
        assert oh[1, 0, 0] == 0.0
        assert oh[0, 0, 1] == 0.0

    def test_single_color(self):
        grid = [[5] * 4 for _ in range(4)]
        oh = one_hot_grid(grid)
        assert oh[5].sum() == 16
        for c in range(MAX_COLORS):
            if c != 5:
                assert oh[c].sum() == 0

    def test_64x64(self, arc_grid):
        oh = one_hot_grid(arc_grid)
        assert oh.shape == (MAX_COLORS, 64, 64)
        # Sum across channels at each position should be 1
        assert np.allclose(oh.sum(axis=0), 1.0)


# ═══════════════════════════════════════════════════════════════════
# TestConvolution
# ═══════════════════════════════════════════════════════════════════


class TestConvolution:
    def test_identity_filter(self):
        """Identity filter should return input unchanged."""
        img = np.random.rand(8, 8).astype(np.float32)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        result = conv2d(img, identity)
        assert result.shape == img.shape
        # Interior should match exactly (edges may differ due to padding)
        np.testing.assert_allclose(result[1:-1, 1:-1], img[1:-1, 1:-1], atol=1e-6)

    def test_output_shape(self):
        img = np.random.rand(64, 64).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        result = conv2d(img, kernel)
        assert result.shape == (64, 64)

    def test_5x5_kernel_shape(self):
        img = np.random.rand(64, 64).astype(np.float32)
        kernel = np.ones((5, 5), dtype=np.float32)
        result = conv2d(img, kernel)
        assert result.shape == (64, 64)

    def test_sobel_detects_edge(self):
        """Sobel filter should produce high values at vertical edges."""
        img = np.zeros((16, 16), dtype=np.float32)
        img[:, 8:] = 1.0  # right half is 1
        sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        result = conv2d(img, sobel_h)
        # Edge at column 7-8 should have high response
        assert abs(result[8, 8]) > abs(result[8, 4])

    def test_uniform_input_zero_laplacian(self):
        """Laplacian of uniform image should be ~zero."""
        img = np.ones((16, 16), dtype=np.float32) * 5.0
        lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        result = conv2d(img, lap)
        # Interior should be zero
        np.testing.assert_allclose(result[2:-2, 2:-2], 0.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════
# TestFilters
# ═══════════════════════════════════════════════════════════════════


class TestFixedFilters:
    def test_shape(self, fixed_filters):
        assert fixed_filters.shape == (FIXED_FILTER_COUNT, 3, 3)

    def test_dtype(self, fixed_filters):
        assert fixed_filters.dtype == np.float32

    def test_identity_is_last(self, fixed_filters):
        """Filter 15 should be identity."""
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(fixed_filters[15], expected)


class TestLearnableFilters:
    def test_shape(self, learnable_filters):
        assert learnable_filters.shape == (LEARNABLE_FILTER_COUNT, 5, 5)

    def test_normalized(self, learnable_filters):
        """Each filter should be L2-normalized."""
        for i in range(LEARNABLE_FILTER_COUNT):
            norm = np.linalg.norm(learnable_filters[i])
            assert abs(norm - 1.0) < 1e-5

    def test_deterministic_seed(self):
        f1 = init_learnable_filters(seed=123)
        f2 = init_learnable_filters(seed=123)
        np.testing.assert_array_equal(f1, f2)

    def test_different_seeds(self):
        f1 = init_learnable_filters(seed=1)
        f2 = init_learnable_filters(seed=2)
        assert not np.allclose(f1, f2)


# ═══════════════════════════════════════════════════════════════════
# TestApplyFilters
# ═══════════════════════════════════════════════════════════════════


class TestApplyFilters:
    def test_output_shape(self, fixed_filters):
        oh = np.random.rand(MAX_COLORS, 16, 16).astype(np.float32)
        result = apply_filters(oh, fixed_filters)
        assert result.shape == (FIXED_FILTER_COUNT, 16, 16)

    def test_relu_no_negatives(self, fixed_filters):
        oh = np.random.rand(MAX_COLORS, 16, 16).astype(np.float32)
        result = apply_filters(oh, fixed_filters)
        assert result.min() >= 0.0

    def test_empty_channels_skipped(self, fixed_filters):
        """Mostly-zero input should be fast and produce sparse output."""
        oh = np.zeros((MAX_COLORS, 16, 16), dtype=np.float32)
        oh[0, 8, 8] = 1.0  # single pixel
        result = apply_filters(oh, fixed_filters)
        # Most of the result should be zero
        assert (result == 0).sum() > result.size * 0.9


# ═══════════════════════════════════════════════════════════════════
# TestFeatureExtraction
# ═══════════════════════════════════════════════════════════════════


class TestFeatureExtraction:
    def test_feature_dim(self, arc_grid, fixed_filters):
        f = extract_features(arc_grid, fixed_filters)
        assert f.shape == (FEATURE_DIM,)

    def test_l2_normalized(self, arc_grid, fixed_filters):
        f = extract_features(arc_grid, fixed_filters)
        assert abs(np.linalg.norm(f) - 1.0) < 1e-5

    def test_dtype(self, arc_grid, fixed_filters):
        f = extract_features(arc_grid, fixed_filters)
        assert f.dtype == np.float32

    def test_identical_grids_same_features(self, arc_grid, fixed_filters):
        f1 = extract_features(arc_grid, fixed_filters)
        f2 = extract_features(arc_grid, fixed_filters)
        np.testing.assert_array_equal(f1, f2)

    def test_different_grids_different_features(self, arc_grid, fixed_filters):
        f1 = extract_features(arc_grid, fixed_filters)
        empty = [[0] * 64 for _ in range(64)]
        f2 = extract_features(empty, fixed_filters)
        assert not np.allclose(f1, f2)

    def test_similar_grids_close_features(self, fixed_filters):
        """Structurally very different grids should produce different features."""
        rng = np.random.RandomState(99)
        # Grid A: uniform single color
        grid_a = [[4] * 64 for _ in range(64)]
        # Grid B: random noise (all 16 colors)
        grid_b = rng.randint(0, 16, size=(64, 64)).tolist()
        f1 = extract_features(grid_a, fixed_filters)
        f2 = extract_features(grid_b, fixed_filters)
        similarity = float(f1 @ f2)
        # Uniform vs random should be very different
        assert similarity < 0.9, f"Uniform vs random should differ, got {similarity}"

    def test_with_learnable_filters(self, arc_grid, fixed_filters, learnable_filters):
        f = extract_features(arc_grid, fixed_filters, learnable_filters)
        assert f.shape == (FEATURE_DIM,)
        assert abs(np.linalg.norm(f) - 1.0) < 1e-5


# ═══════════════════════════════════════════════════════════════════
# TestGridHash
# ═══════════════════════════════════════════════════════════════════


class TestGridHash:
    def test_length(self, arc_grid):
        h = grid_hash(arc_grid)
        assert len(h) == 12

    def test_deterministic(self, arc_grid):
        h1 = grid_hash(arc_grid)
        h2 = grid_hash(arc_grid)
        assert h1 == h2

    def test_different_grids_different_hash(self, arc_grid):
        h1 = grid_hash(arc_grid)
        empty = [[0] * 64 for _ in range(64)]
        h2 = grid_hash(empty)
        assert h1 != h2

    def test_hex_string(self, arc_grid):
        h = grid_hash(arc_grid)
        int(h, 16)  # should not raise


# ═══════════════════════════════════════════════════════════════════
# TestPerceptionIndex
# ═══════════════════════════════════════════════════════════════════


class TestPerceptionIndex:
    def test_empty_query(self):
        idx = PerceptionIndex()
        result = idx.query(np.zeros(FEATURE_DIM), top_k=5)
        assert result == []

    def test_add_and_query(self):
        idx = PerceptionIndex()
        v1 = np.random.randn(FEATURE_DIM).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        idx.add("a", v1)
        assert idx.size == 1

        results = idx.query(v1, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "a"
        assert results[0][1] > 0.99

    def test_build(self):
        idx = PerceptionIndex()
        features = {}
        for i in range(10):
            v = np.random.randn(FEATURE_DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            features[f"frame_{i}"] = v
        idx.build(features)
        assert idx.size == 10

    def test_query_returns_sorted(self):
        idx = PerceptionIndex()
        # Add target and noise
        target = np.ones(FEATURE_DIM, dtype=np.float32)
        target /= np.linalg.norm(target)
        idx.add("target", target)

        for i in range(5):
            v = np.random.randn(FEATURE_DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            idx.add(f"noise_{i}", v)

        results = idx.query(target, top_k=3)
        assert results[0][0] == "target"
        # Similarities should be descending
        sims = [r[1] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_remove(self):
        idx = PerceptionIndex()
        v = np.ones(FEATURE_DIM, dtype=np.float32)
        v /= np.linalg.norm(v)
        idx.add("a", v)
        assert idx.size == 1
        assert idx.remove("a")
        assert idx.size == 0
        assert not idx.remove("a")  # already removed

    def test_update_existing(self):
        idx = PerceptionIndex()
        v1 = np.ones(FEATURE_DIM, dtype=np.float32)
        v1 /= np.linalg.norm(v1)
        idx.add("a", v1)

        v2 = -v1  # opposite direction
        idx.add("a", v2)
        assert idx.size == 1  # should not duplicate

        results = idx.query(v2, top_k=1)
        assert results[0][1] > 0.99  # should match updated vector


# ═══════════════════════════════════════════════════════════════════
# TestHebbianLearning
# ═══════════════════════════════════════════════════════════════════


class TestHebbianLearning:
    def test_zero_reward_no_change(self, learnable_filters):
        oh = np.random.rand(MAX_COLORS, 16, 16).astype(np.float32)
        act = apply_filters(oh, learnable_filters)
        updated = update_filters_hebbian(learnable_filters, oh, act, reward=0.0)
        np.testing.assert_array_equal(updated, learnable_filters)

    def test_positive_reward_changes_filters(self, learnable_filters):
        oh = np.random.rand(MAX_COLORS, 16, 16).astype(np.float32)
        act = apply_filters(oh, learnable_filters)
        updated = update_filters_hebbian(learnable_filters, oh, act, reward=1.0)
        assert not np.allclose(updated, learnable_filters)

    def test_filters_stay_normalized(self, learnable_filters):
        oh = np.random.rand(MAX_COLORS, 16, 16).astype(np.float32)
        act = apply_filters(oh, learnable_filters)
        updated = update_filters_hebbian(
            learnable_filters, oh, act, reward=1.0, learning_rate=0.1
        )
        for i in range(LEARNABLE_FILTER_COUNT):
            norm = np.linalg.norm(updated[i])
            assert abs(norm - 1.0) < 1e-5, f"Filter {i} norm = {norm}"

    def test_negative_reward_changes_differently(self, learnable_filters):
        oh = np.random.rand(MAX_COLORS, 16, 16).astype(np.float32)
        act = apply_filters(oh, learnable_filters)
        pos = update_filters_hebbian(learnable_filters, oh, act, reward=1.0)
        neg = update_filters_hebbian(learnable_filters, oh, act, reward=-1.0)
        # Positive and negative updates should go in different directions
        assert not np.allclose(pos, neg)


# ═══════════════════════════════════════════════════════════════════
# TestPerformance
# ═══════════════════════════════════════════════════════════════════


class TestPerformance:
    @pytest.mark.slow
    def test_extract_features_speed(self, arc_grid, fixed_filters):
        """Feature extraction should be < 50ms per frame."""
        import time

        t0 = time.perf_counter()
        for _ in range(10):
            extract_features(arc_grid, fixed_filters)
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.050, f"Too slow: {elapsed*1000:.1f}ms per frame"

    @pytest.mark.slow
    def test_index_query_speed(self):
        """Index query on 1000 entries should be < 5ms."""
        import time

        idx = PerceptionIndex()
        rng = np.random.RandomState(42)
        for i in range(1000):
            v = rng.randn(FEATURE_DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            idx.add(f"f_{i}", v)

        query = rng.randn(FEATURE_DIM).astype(np.float32)
        query /= np.linalg.norm(query)

        t0 = time.perf_counter()
        for _ in range(100):
            idx.query(query, top_k=5)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005, f"Too slow: {elapsed*1000:.1f}ms per query"
