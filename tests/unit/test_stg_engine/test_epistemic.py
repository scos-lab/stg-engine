"""Tests for Phase 7H: Epistemic Metadata Layer.

Tests edge classification, epistemic metadata validation,
query filtering, summary, persistence round-trip, and importer tagging.
"""

import os
import tempfile
import pytest

from stg_engine.engine import STGEngine
from stg_engine.epistemic import (
    get_edge_class,
    validate_epistemic_metadata,
    epistemic_summary,
    VALID_EDGE_CLASSES,
    VALID_TRACE_TYPES,
    VALID_VERIFICATION_STATUSES,
    VALID_EPISTEMIC_STATUSES,
    VALID_SCOPES,
    EPISTEMIC_KEYS,
    CONFIDENCE_RANGES,
)
from stg_engine.persistence import save_engine_state, load_engine_state


# ─── Helpers ──────────────────────────────────────────────────────

def make_engine_with_edges():
    """Create an engine with various edge types for testing."""
    engine = STGEngine()

    # Structural edges (default)
    engine.add_edge("doc:readme", "section:intro", confidence=0.95,
                    edge_class="structural")
    engine.add_edge("class:Engine", "method:query", confidence=0.90,
                    edge_class="structural")

    # Cognitive edges
    engine.add_edge("Plan_A", "Goal_Performance", confidence=0.80,
                    edge_class="cognitive")
    engine.add_edge("Observation_Speed", "Conclusion_NoRust", confidence=0.85,
                    edge_class="cognitive")

    # Knowledge edges with full epistemic metadata
    engine.add_edge("QFT", "Fields_Fundamental", confidence=0.97,
                    edge_class="knowledge",
                    trace_type="EarthTrace",
                    structural_coherence=0.99,
                    verification_status="consensus",
                    epistemic_status="established",
                    scope="physical")

    engine.add_edge("Bohm", "Implicate_Order", confidence=0.65,
                    edge_class="knowledge",
                    trace_type="CosmicTrace",
                    structural_coherence=0.92,
                    verification_status="beyond_current_paradigm",
                    epistemic_status="speculative",
                    scope="physical")

    engine.add_edge("TCM_Meridian", "Clinical_Effect", confidence=0.72,
                    edge_class="knowledge",
                    trace_type="CosmicTrace",
                    structural_coherence=0.87,
                    verification_status="reproducible",
                    epistemic_status="provisional",
                    scope="cultural")

    engine.add_edge("Meditation_Report", "Inner_Light", confidence=0.50,
                    edge_class="knowledge",
                    trace_type="UserClaimed",
                    verification_status="subjective",
                    epistemic_status="speculative",
                    scope="personal")

    # Legacy edge (no edge_class — defaults to structural)
    engine.add_edge("A", "B", confidence=0.5)

    return engine


# ═══════════════════════════════════════════════════════════════════
# Class 1: Edge Class Types
# ═══════════════════════════════════════════════════════════════════

class TestEdgeClassTypes:
    def test_valid_edge_classes(self):
        assert VALID_EDGE_CLASSES == {"cognitive", "knowledge", "structural", "virtual"}

    def test_invalid_edge_class_warning(self):
        warnings = validate_epistemic_metadata(0.5, {"edge_class": "unknown"})
        assert any("Unknown edge_class" in w for w in warnings)

    def test_default_edge_class(self):
        assert get_edge_class({}) == "structural"
        assert get_edge_class({"edge_class": "cognitive"}) == "cognitive"
        assert get_edge_class({"edge_class": "knowledge"}) == "knowledge"
        assert get_edge_class({"edge_class": "structural"}) == "structural"

    def test_edge_class_stored_in_modifiers(self):
        engine = STGEngine()
        edge = engine.add_edge("A", "B", edge_class="knowledge")
        assert edge.modifiers["edge_class"] == "knowledge"
        assert edge.edge_class == "knowledge"

    def test_all_three_classes_accepted(self):
        """All three valid classes produce no edge_class warnings."""
        for cls in VALID_EDGE_CLASSES:
            warnings = validate_epistemic_metadata(0.5, {"edge_class": cls})
            assert not any("Unknown edge_class" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════
# Class 2: Trace Types
# ═══════════════════════════════════════════════════════════════════

class TestTraceTypes:
    def test_valid_trace_types(self):
        assert VALID_TRACE_TYPES == {"EarthTrace", "CosmicTrace", "UserClaimed"}

    def test_invalid_trace_type_warning(self):
        warnings = validate_epistemic_metadata(0.5, {
            "edge_class": "knowledge",
            "trace_type": "InvalidType",
        })
        assert any("Unknown trace_type" in w for w in warnings)

    def test_hallucination_not_stored(self):
        """There is no 'Hallucination' trace type — hallucinations are rejected, not stored."""
        assert "Hallucination" not in VALID_TRACE_TYPES

    def test_each_trace_type_valid(self):
        for tt in VALID_TRACE_TYPES:
            warnings = validate_epistemic_metadata(0.65, {
                "edge_class": "knowledge",
                "trace_type": tt,
                "structural_coherence": 0.80,
            })
            assert not any("Unknown trace_type" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════
# Class 3: Validation Rules
# ═══════════════════════════════════════════════════════════════════

class TestValidationRules:
    def test_cosmic_trace_requires_coherence(self):
        warnings = validate_epistemic_metadata(0.65, {
            "edge_class": "knowledge",
            "trace_type": "CosmicTrace",
        })
        assert any("structural_coherence specified" in w for w in warnings)

    def test_cosmic_trace_low_coherence_warning(self):
        warnings = validate_epistemic_metadata(0.65, {
            "edge_class": "knowledge",
            "trace_type": "CosmicTrace",
            "structural_coherence": 0.55,
        })
        assert any("structural_coherence >= 0.70" in w for w in warnings)

    def test_cosmic_trace_valid_coherence(self):
        warnings = validate_epistemic_metadata(0.65, {
            "edge_class": "knowledge",
            "trace_type": "CosmicTrace",
            "structural_coherence": 0.85,
        })
        assert not any("structural_coherence" in w for w in warnings)

    def test_confidence_range_earth_trace(self):
        # Below range
        w = validate_epistemic_metadata(0.40, {
            "edge_class": "knowledge",
            "trace_type": "EarthTrace",
        })
        assert any("[0.6, 0.98]" in w for w in w)

        # Above range
        w2 = validate_epistemic_metadata(1.0, {
            "edge_class": "knowledge",
            "trace_type": "EarthTrace",
        })
        assert any("[0.6, 0.98]" in w for w in w2)

    def test_confidence_range_cosmic_trace(self):
        w = validate_epistemic_metadata(0.90, {
            "edge_class": "knowledge",
            "trace_type": "CosmicTrace",
            "structural_coherence": 0.80,
        })
        assert any("[0.5, 0.75]" in w for w in w)

    def test_confidence_range_user_claimed(self):
        w = validate_epistemic_metadata(0.90, {
            "edge_class": "knowledge",
            "trace_type": "UserClaimed",
        })
        assert any("[0.35, 0.7]" in w for w in w)

    def test_epistemic_on_structural_warns(self):
        warnings = validate_epistemic_metadata(0.5, {
            "edge_class": "structural",
            "trace_type": "EarthTrace",
        })
        assert any("designed for knowledge" in w for w in warnings)

    def test_epistemic_on_cognitive_warns(self):
        warnings = validate_epistemic_metadata(0.5, {
            "edge_class": "cognitive",
            "trace_type": "EarthTrace",
        })
        assert any("designed for knowledge" in w for w in warnings)

    def test_valid_knowledge_edge_no_warnings(self):
        warnings = validate_epistemic_metadata(0.90, {
            "edge_class": "knowledge",
            "trace_type": "EarthTrace",
            "structural_coherence": 0.95,
            "verification_status": "consensus",
            "epistemic_status": "established",
            "scope": "physical",
        })
        assert warnings == []

    def test_structural_coherence_range(self):
        w = validate_epistemic_metadata(0.65, {
            "edge_class": "knowledge",
            "trace_type": "CosmicTrace",
            "structural_coherence": 1.5,
        })
        assert any("[0.0, 1.0]" in w for w in w)

    def test_invalid_verification_status_warning(self):
        w = validate_epistemic_metadata(0.80, {
            "edge_class": "knowledge",
            "trace_type": "EarthTrace",
            "verification_status": "invalid_status",
        })
        assert any("Unknown verification_status" in w for w in w)

    def test_invalid_epistemic_status_warning(self):
        w = validate_epistemic_metadata(0.80, {
            "edge_class": "knowledge",
            "trace_type": "EarthTrace",
            "epistemic_status": "invalid_status",
        })
        assert any("Unknown epistemic_status" in w for w in w)

    def test_invalid_scope_warning(self):
        w = validate_epistemic_metadata(0.80, {
            "edge_class": "knowledge",
            "trace_type": "EarthTrace",
            "scope": "invalid_scope",
        })
        assert any("Unknown scope" in w for w in w)

    def test_all_verification_statuses_valid(self):
        for vs in VALID_VERIFICATION_STATUSES:
            w = validate_epistemic_metadata(0.80, {
                "edge_class": "knowledge",
                "trace_type": "EarthTrace",
                "verification_status": vs,
            })
            assert not any("Unknown verification_status" in w for w in w)

    def test_all_epistemic_statuses_valid(self):
        for es in VALID_EPISTEMIC_STATUSES:
            w = validate_epistemic_metadata(0.80, {
                "edge_class": "knowledge",
                "trace_type": "EarthTrace",
                "epistemic_status": es,
            })
            assert not any("Unknown epistemic_status" in w for w in w)

    def test_all_scopes_valid(self):
        for sc in VALID_SCOPES:
            w = validate_epistemic_metadata(0.80, {
                "edge_class": "knowledge",
                "trace_type": "EarthTrace",
                "scope": sc,
            })
            assert not any("Unknown scope" in w for w in w)


# ═══════════════════════════════════════════════════════════════════
# Class 4: Query Edges with Epistemic Filters
# ═══════════════════════════════════════════════════════════════════

class TestQueryEdgesEpistemic:
    def test_filter_by_edge_class(self):
        engine = make_engine_with_edges()
        cognitive = engine.query_edges(edge_class="cognitive")
        assert len(cognitive) == 2
        for e in cognitive:
            assert e.modifiers.get("edge_class") == "cognitive"

    def test_filter_by_trace_type(self):
        engine = make_engine_with_edges()
        earth = engine.query_edges(trace_type="EarthTrace")
        assert len(earth) == 1
        assert earth[0].source == "QFT"

    def test_filter_by_verification_status(self):
        engine = make_engine_with_edges()
        consensus = engine.query_edges(verification_status="consensus")
        assert len(consensus) == 1
        assert consensus[0].source == "QFT"

    def test_filter_by_epistemic_status(self):
        engine = make_engine_with_edges()
        speculative = engine.query_edges(epistemic_status="speculative")
        assert len(speculative) == 2  # Bohm + Meditation

    def test_filter_by_scope(self):
        engine = make_engine_with_edges()
        physical = engine.query_edges(scope="physical")
        assert len(physical) == 2  # QFT + Bohm

    def test_filter_by_min_structural_coherence(self):
        engine = make_engine_with_edges()
        high_sc = engine.query_edges(min_structural_coherence=0.90)
        assert len(high_sc) == 2  # QFT (0.99) + Bohm (0.92)

    def test_combined_filters(self):
        engine = make_engine_with_edges()
        results = engine.query_edges(
            trace_type="CosmicTrace",
            scope="physical",
        )
        assert len(results) == 1  # Only Bohm
        assert results[0].source == "Bohm"

    def test_epistemic_filters_with_existing_filters(self):
        engine = make_engine_with_edges()
        results = engine.query_edges(
            min_confidence=0.70,
            edge_class="knowledge",
        )
        # QFT (0.97) + TCM (0.72) — Bohm is 0.65, Meditation is 0.50
        assert len(results) == 2

    def test_default_edge_class_structural(self):
        """Legacy edges (no edge_class) are found by edge_class='structural'."""
        engine = make_engine_with_edges()
        structural = engine.query_edges(edge_class="structural")
        # 2 explicit structural + 1 legacy (A->B)
        assert len(structural) == 3


# ═══════════════════════════════════════════════════════════════════
# Class 5: Epistemic Summary
# ═══════════════════════════════════════════════════════════════════

class TestEpistemicSummary:
    def test_summary_edge_class_distribution(self):
        engine = make_engine_with_edges()
        s = engine.epistemic_summary()
        assert s["edge_class_distribution"]["structural"] == 3  # 2 explicit + 1 legacy
        assert s["edge_class_distribution"]["cognitive"] == 2
        assert s["edge_class_distribution"]["knowledge"] == 4

    def test_summary_trace_type_distribution(self):
        engine = make_engine_with_edges()
        s = engine.epistemic_summary()
        assert s["trace_type_distribution"]["EarthTrace"] == 1
        assert s["trace_type_distribution"]["CosmicTrace"] == 2
        assert s["trace_type_distribution"]["UserClaimed"] == 1

    def test_summary_empty_graph(self):
        engine = STGEngine()
        s = engine.epistemic_summary()
        assert s["total_edge_count"] == 0
        assert s["knowledge_edge_count"] == 0
        assert s["edge_class_distribution"]["structural"] == 0

    def test_summary_all_structural(self):
        engine = STGEngine()
        engine.add_edge("A", "B")
        engine.add_edge("C", "D")
        s = engine.epistemic_summary()
        assert s["total_edge_count"] == 2
        assert s["edge_class_distribution"]["structural"] == 2
        assert s["knowledge_edge_count"] == 0

    def test_summary_knowledge_edge_count(self):
        engine = make_engine_with_edges()
        s = engine.epistemic_summary()
        assert s["knowledge_edge_count"] == 4

    def test_summary_scope_distribution(self):
        engine = make_engine_with_edges()
        s = engine.epistemic_summary()
        assert s["scope_distribution"]["physical"] == 2
        assert s["scope_distribution"]["cultural"] == 1
        assert s["scope_distribution"]["personal"] == 1


# ═══════════════════════════════════════════════════════════════════
# Class 6: Importer Tagging
# ═══════════════════════════════════════════════════════════════════

class TestImporterTagging:
    def test_add_edge_with_edge_class(self):
        engine = STGEngine()
        edge = engine.add_edge("A", "B", edge_class="knowledge")
        assert edge.modifiers["edge_class"] == "knowledge"

    def test_add_edge_default_no_class(self):
        engine = STGEngine()
        edge = engine.add_edge("A", "B")
        assert "edge_class" not in edge.modifiers

    def test_add_edge_with_full_epistemic(self):
        engine = STGEngine()
        edge = engine.add_edge("QFT", "Fields", confidence=0.97,
                               edge_class="knowledge",
                               trace_type="EarthTrace",
                               structural_coherence=0.99,
                               verification_status="consensus",
                               epistemic_status="established",
                               scope="physical")
        assert edge.modifiers["edge_class"] == "knowledge"
        assert edge.modifiers["trace_type"] == "EarthTrace"
        assert edge.modifiers["structural_coherence"] == 0.99
        assert edge.modifiers["verification_status"] == "consensus"
        assert edge.modifiers["epistemic_status"] == "established"
        assert edge.modifiers["scope"] == "physical"

    def test_validation_warnings_stored_in_edge(self):
        """Validation warnings are stored in edge.modifiers['_epistemic_warnings']."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.30,
                               edge_class="knowledge",
                               trace_type="EarthTrace")
        # confidence=0.30 is below EarthTrace range (0.60-0.98)
        assert "_epistemic_warnings" in edge.modifiers
        assert any("typically in" in w for w in edge.modifiers["_epistemic_warnings"])

    def test_no_warnings_for_valid_edge(self):
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.90,
                               edge_class="knowledge",
                               trace_type="EarthTrace",
                               structural_coherence=0.95)
        assert "_epistemic_warnings" not in edge.modifiers


# ═══════════════════════════════════════════════════════════════════
# Class 7: Persistence Round-Trip
# ═══════════════════════════════════════════════════════════════════

class TestPersistenceRoundTrip:
    def test_epistemic_metadata_survives_save_load(self):
        """Save .stg with epistemic metadata, reload, verify all fields preserved."""
        engine = STGEngine()
        engine.add_edge("QFT", "Fields", confidence=0.97,
                        edge_class="knowledge",
                        trace_type="EarthTrace",
                        structural_coherence=0.99,
                        verification_status="consensus",
                        epistemic_status="established",
                        scope="physical")
        engine.add_edge("Plan_A", "Goal_X", confidence=0.80,
                        edge_class="cognitive")

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name

        try:
            engine.save(path)
            loaded = STGEngine.load(path)

            # Check knowledge edge
            knowledge = loaded.query_edges(edge_class="knowledge")
            assert len(knowledge) == 1
            e = knowledge[0]
            assert e.modifiers["edge_class"] == "knowledge"
            assert e.modifiers["trace_type"] == "EarthTrace"
            assert e.modifiers["structural_coherence"] == 0.99
            assert e.modifiers["verification_status"] == "consensus"
            assert e.modifiers["epistemic_status"] == "established"
            assert e.modifiers["scope"] == "physical"

            # Check cognitive edge
            cognitive = loaded.query_edges(edge_class="cognitive")
            assert len(cognitive) == 1
            assert cognitive[0].modifiers["edge_class"] == "cognitive"
        finally:
            os.unlink(path)

    def test_legacy_stg_loads_without_epistemic(self):
        """Old .stg files without epistemic metadata load without errors."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        engine.add_edge("C", "D", confidence=0.8, rule="causal")

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name

        try:
            engine.save(path)
            loaded = STGEngine.load(path)

            # All edges load fine — no edge_class in modifiers
            assert len(loaded._edges) == 2
            for e in loaded._edges:
                # Default to structural
                assert get_edge_class(e.modifiers) == "structural"

            # Summary works on legacy data
            s = loaded.epistemic_summary()
            assert s["total_edge_count"] == 2
            assert s["edge_class_distribution"]["structural"] == 2
        finally:
            os.unlink(path)
