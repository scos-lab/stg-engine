"""Universal knowledge base importer for STG Engine.

Manifest-driven orchestrator that imports the full knowledge ecosystem
into the STG: concept skeleton, STL-native files, STLC specifications,
and generic markdown documents.

Supports both STL and JSON manifest formats:
    - stg_manifest.stl (preferred) — STL statements with path/type/namespace/priority modifiers
    - stg_manifest.json (legacy)   — JSON array of source objects

Usage:
    from stg_engine import import_knowledge_base
    engine = import_knowledge_base("stg_manifest.stl", project_root="/path/to/skc")

Or via CLI:
    python stg_cli.py import
"""

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

from stg_engine.engine import STGEngine
from stg_engine.concept_skeleton import inject_skeleton, CORE_CONCEPT_NAMES
from stg_engine.importers import import_memory_matrix, _ingest_all_stl
from stg_engine.markdown_extractor import extract_markdown_structure
from stg_engine.spec_extractor import extract_spec


def import_knowledge_base(
    manifest_path: Optional[str] = None,
    engine: Optional[STGEngine] = None,
    project_root: Optional[str] = None,
) -> STGEngine:
    """Import full knowledge base into STG engine.

    Phases:
        0. Inject concept skeleton (hub nodes + edges)
        1. Load manifest and resolve source paths
        2. Import sources by priority (1=core, 2=spec, 3=doc)
        3. Compute tensions and activations

    Args:
        manifest_path: Path to stg_manifest.stl or stg_manifest.json.
            Default: <project_root>/stg_manifest.stl (falls back to .json)
        engine: Optional existing engine. Creates new if None.
        project_root: SKC project root directory.
            Default: directory containing manifest.

    Returns:
        STGEngine with all knowledge imported
    """
    if engine is None:
        engine = STGEngine()

    # Resolve project root
    if project_root is None and manifest_path:
        project_root = os.path.dirname(os.path.abspath(manifest_path))
    elif project_root is None:
        project_root = os.getcwd()

    # Resolve manifest path — prefer .stl over .json
    if manifest_path is None:
        stl_path = os.path.join(project_root, "stg_manifest.stl")
        json_path = os.path.join(project_root, "stg_manifest.json")
        if os.path.isfile(stl_path):
            manifest_path = stl_path
        else:
            manifest_path = json_path

    # Phase 0: Inject concept skeleton
    skeleton_count = inject_skeleton(engine)

    # Phase 1: Load manifest
    manifest = _load_manifest(manifest_path)

    # Phase 2: Import sources by priority
    sources = sorted(manifest.get("sources", []), key=lambda s: s.get("priority", 99))
    import_stats: Dict[str, int] = {
        "skeleton": skeleton_count,
        "stl_native": 0,
        "stlc_spec": 0,
        "markdown_doc": 0,
        "files_processed": 0,
        "files_skipped": 0,
    }

    for source in sources:
        source_type = source.get("type", "markdown_doc")
        namespace = source.get("namespace", "Doc")
        path_pattern = source["path"]

        # Resolve paths (handle globs)
        resolved_paths = _resolve_paths(path_pattern, project_root)

        for filepath in resolved_paths:
            if not os.path.isfile(filepath):
                import_stats["files_skipped"] += 1
                continue

            try:
                if source_type == "stl_native":
                    count = _import_stl_native(filepath, engine, namespace, project_root)
                    import_stats["stl_native"] += count
                elif source_type == "stlc_spec":
                    count = extract_spec(filepath, engine, namespace, project_root)
                    import_stats["stlc_spec"] += count
                elif source_type == "markdown_doc":
                    count = extract_markdown_structure(filepath, engine, namespace, project_root)
                    import_stats["markdown_doc"] += count
                else:
                    # Unknown type, treat as markdown
                    count = extract_markdown_structure(filepath, engine, namespace, project_root)
                    import_stats["markdown_doc"] += count

                import_stats["files_processed"] += 1
            except Exception as e:
                import_stats["files_skipped"] += 1
                # Continue importing other files even if one fails

    # Phase 2.5: Materialize tensions as graph nodes
    tension_count = _materialize_tensions(engine)
    import_stats["tension_nodes"] = tension_count

    # Phase 3: Compute tensions and activations
    engine.compute_all_tensions()
    engine.compute_activations()

    # Phase 3.5: Compute self-relevance (distance from Self node)
    _compute_self_relevance(engine)

    # Store import stats as engine metadata
    engine._import_stats = import_stats

    return engine


def _load_manifest(path: str) -> Dict[str, Any]:
    """Load manifest from .stl or .json format.

    STL format: each statement's modifiers contain path, type, namespace, priority.
    JSON format: {"sources": [{"path": ..., "type": ..., ...}]}
    """
    if path.endswith(".stl"):
        return _load_stl_manifest(path)

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "sources" not in manifest:
        raise ValueError(f"Manifest missing 'sources' key: {path}")

    return manifest


def _load_stl_manifest(path: str) -> Dict[str, Any]:
    """Parse STL manifest into the same dict structure as JSON manifest.

    Uses stl_parser to parse the file — dogfooding STL's own toolchain.

    Each STL statement like:
        [STG_Manifest] -> [Source:name] ::mod(path="...", type="...", namespace="...", priority=1)
    becomes a source entry: {"path": "...", "type": "...", "namespace": "...", "priority": 1}
    """
    from stl_parser import parse_file

    result = parse_file(path)
    sources: List[Dict[str, Any]] = []

    for stmt in result.statements:
        mods = stmt.modifiers
        if not mods:
            continue

        # Extract manifest-relevant fields from modifiers
        source_entry = _modifiers_to_dict(mods)

        if "path" in source_entry:
            # Ensure priority is int
            if "priority" in source_entry:
                try:
                    source_entry["priority"] = int(source_entry["priority"])
                except (ValueError, TypeError):
                    source_entry["priority"] = 3
            sources.append(source_entry)

    if not sources:
        raise ValueError(f"No source entries found in STL manifest: {path}")

    return {"sources": sources}


def _modifiers_to_dict(mods: Any) -> Dict[str, Any]:
    """Extract all non-None modifier fields into a plain dict.

    Works with stl_parser's Modifiers object — reads both standard
    fields (priority, etc.) and dynamic attributes (path, type, namespace).
    """
    result: Dict[str, Any] = {}

    # Manifest fields we care about
    for key in ("path", "type", "namespace", "priority", "description"):
        val = getattr(mods, key, None)
        if val is not None:
            result[key] = val

    # Also include any custom fields
    custom = getattr(mods, "custom", {})
    if custom:
        result.update(custom)

    return result


def _resolve_paths(pattern: str, project_root: str) -> List[str]:
    """Resolve a path pattern (possibly with globs) to actual file paths.

    Supports:
    - Exact paths: 'memory/Syn-claude/memoryMatrix.md'
    - Glob patterns: 'development/specifications/**/*.md'
    - Paths starting with '../' for files outside project root
    """
    # Handle absolute paths
    if os.path.isabs(pattern):
        if os.path.isfile(pattern):
            return [pattern]
        return glob.glob(pattern, recursive=True)

    # Make relative to project root
    full_pattern = os.path.join(project_root, pattern)

    # Normalize path separators for Windows
    full_pattern = full_pattern.replace("/", os.sep)

    if "*" in pattern or "?" in pattern:
        # Glob pattern
        paths = glob.glob(full_pattern, recursive=True)
        return sorted(paths)
    elif os.path.isfile(full_pattern):
        return [full_pattern]
    else:
        return []


def _import_stl_native(
    filepath: str,
    engine: STGEngine,
    namespace: str,
    project_root: Optional[str] = None,
) -> int:
    """Import an STL-native file (contains STL statements).

    For memoryMatrix.md, uses the full structured importer.
    For other files, creates a document node + ingests STL blocks.
    """
    basename = os.path.basename(filepath).lower()
    count = 0

    if basename == "memorymatrix.md":
        # Use full memoryMatrix importer (sessions, events, tensions, etc.)
        import_memory_matrix(filepath, engine)
        count = engine.get_stats()["node_count"]  # Approximate
    else:
        # Create document node
        from stg_engine.markdown_extractor import _derive_doc_name
        doc_name = _derive_doc_name(filepath)
        doc_node_name = f"{namespace}:{doc_name}"

        rel_path = filepath
        if project_root:
            try:
                rel_path = os.path.relpath(filepath, project_root)
            except ValueError:
                pass

        # Extract title
        title = doc_name
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        lines = content.split('\n')
        for line in lines[:20]:
            import re
            m = re.match(r'^#\s+(.+)', line)
            if m:
                title = m.group(1).strip()
                break

        engine.add_node(
            doc_node_name,
            namespace=namespace,
            anchor_type="Entity",
            file=rel_path,
            lines=f"1-{len(lines)}",
            title=title,
        )
        count += 1

        # Ingest all STL statements from the file
        stl_count = _ingest_all_stl(content, engine)
        count += stl_count

        # Also extract structural sections (like markdown_extractor does)
        struct_count = extract_markdown_structure(filepath, engine, namespace, project_root)
        count += struct_count

    return count


def _materialize_tensions(engine: STGEngine) -> int:
    """Create graph nodes for tensions and connect to related concepts.

    Tensions exist in the tensions table but are invisible to graph operations
    (paths, propagate). This function materializes them as nodes with edges
    to their related concept nodes, unifying the tension and graph worlds.

    Returns:
        Number of elements added (nodes + edges)
    """
    tensions = engine.query_tensions()
    if not tensions:
        return 0

    count = 0
    for tension in tensions:
        # Create tension node in Tension namespace
        node_name = f"Tension:{tension.name}"
        engine.add_node(
            node_name,
            namespace="Tension",
            anchor_type="Question",
            tension_status=tension.status,
            tension_value=str(tension.current_value),
            description=tension.description or "",
        )
        count += 1

        # Find related concept nodes by scanning tension name
        related = _find_related_concepts(tension.name, engine)
        for concept_name in related:
            engine.add_edge(
                node_name, concept_name,
                confidence=min(tension.current_value + 0.2, 1.0),
                strength=0.75,
                rule="tension",
                type="tension_of",
                edge_class="structural",
            )
            count += 1

        # Connect to session if available
        if tension.created_session:
            session_nodes = engine.query_nodes(
                name_pattern=tension.created_session, limit=1
            )
            if session_nodes:
                engine.add_edge(
                    session_nodes[0].name, node_name,
                    confidence=0.90,
                    strength=0.6,
                    rule="causal",
                    type="created_tension",
                    edge_class="structural",
                )
                count += 1

    return count


def _find_related_concepts(tension_name: str, engine: STGEngine) -> List[str]:
    """Find concept nodes related to a tension by name analysis.

    Scans tension name for known concept names. Also handles
    arrow notation like 'Self→Self_Anchor_Definition'.
    """
    related = []

    # Normalize: replace arrows and split into parts
    normalized = tension_name.replace("→", " ").replace("->", " ")

    for concept_name in CORE_CONCEPT_NAMES:
        # Check if concept name appears in the tension name
        pattern = concept_name.replace("_", r"[\s_]")
        if re.search(pattern, normalized, re.IGNORECASE):
            # Verify the concept node exists in the graph
            if engine.get_node(concept_name):
                related.append(concept_name)

    # If no concepts matched, try matching individual words to node names
    if not related:
        words = re.split(r'[_\s→\->]+', tension_name)
        for word in words:
            if len(word) < 3:
                continue
            matches = engine.query_nodes(name_pattern=word, limit=3)
            for m in matches:
                # Prefer skeleton/concept nodes over document nodes
                if ":" not in m.name and m.name not in related:
                    related.append(m.name)
                    break

    return related


def _compute_self_relevance(engine: STGEngine) -> None:
    """Compute self-relevance for all nodes based on distance from Self.

    self_relevance(node) = 1.0 / (1 + shortest_path_distance(Self, node))

    Uses BFS on the undirected view of the graph for reachability.
    Nodes unreachable from Self get self_relevance = 0.0.
    """
    self_node = engine.get_node("Self")
    if not self_node:
        return

    # BFS from Self on undirected graph
    graph = engine._graph
    undirected = graph.to_undirected()

    import networkx as nx
    try:
        distances = dict(
            nx.single_source_shortest_path_length(undirected, "Self")
        )
    except Exception:
        return

    # Set self_relevance on all nodes
    for name, node in engine._nodes.items():
        if name in distances:
            node.self_relevance = 1.0 / (1.0 + distances[name])
        # else: stays 0.0 (unreachable from Self)
