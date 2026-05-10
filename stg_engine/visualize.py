"""STG 3D Star Map Visualization — Phase 9V.

Renders the Semantic Tension Graph as an interactive 3D star map
in the browser. Each node is a luminous point; edges are thin lines.
Semantically similar nodes cluster together via UMAP or PCA.

Output: a single self-contained HTML file using 3d-force-graph (CDN).
No server, no build step — just open in browser.
"""

import json
import math
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine

# ═══════════════════════════════════════════════════════════
# Namespace color palette
# ═══════════════════════════════════════════════════════════

NAMESPACE_COLORS: Dict[str, str] = {
    "STL": "#FFD700",
    "Identity": "#00FFFF",
    "Foundation": "#9B59B6",
    "Memory": "#3498DB",
    "Spec": "#2ECC71",
    "Archive": "#7F8C8D",
    "Research": "#E67E22",
    "Meta": "#ECF0F1",
    "Reference": "#BDC3C7",
    "Cortex": "#E91E63",
    "Doc": "#1ABC9C",
}

RULE_COLORS: Dict[str, str] = {
    "causal": "#E74C3C",
    "logical": "#3498DB",
    "empirical": "#2ECC71",
    "definitional": "#95A5A6",
}

DEFAULT_NODE_COLOR = "#85C1E9"
DEFAULT_EDGE_COLOR = "#444444"


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════


@dataclass
class VisualizationConfig:
    """Configuration for 3D star map visualization."""
    mode: str = "full"                # "full" | "ego" | "community" | "propagate"
    ego_node: Optional[str] = None
    ego_depth: int = 2
    community_id: Optional[int] = None
    propagate_query: Optional[str] = None
    use_embeddings: bool = True
    random_state: int = 42
    background_color: str = "#000011"
    node_size_min: float = 1.0
    node_size_max: float = 8.0
    edge_opacity: float = 0.15
    label_threshold: int = 5
    output_path: Optional[str] = None
    auto_open: bool = True
    filter_dead_ends: bool = False


# ═══════════════════════════════════════════════════════════
# Coordinate Engine
# ═══════════════════════════════════════════════════════════


class CoordinateEngine:
    """Manages 3D coordinate computation from embeddings."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def compute_coordinates(
        self,
        engine: "STGEngine",
    ) -> Dict[str, Tuple[float, float, float]]:
        """Compute 3D coordinates for all nodes with embeddings.

        Uses UMAP if available, falls back to PCA, then random.
        """
        if engine._vector_index is None or engine._vector_index.size == 0:
            return {}

        names = list(engine._vector_index.names)
        matrix = engine._vector_index.matrix  # (N, 384)

        if len(names) < 3:
            return {n: (0.0, 0.0, 0.0) for n in names}

        coords_3d = self._reduce_to_3d(matrix)
        coords_3d = self._normalize(coords_3d)

        return {
            names[i]: (float(coords_3d[i, 0]), float(coords_3d[i, 1]), float(coords_3d[i, 2]))
            for i in range(len(names))
        }

    def _reduce_to_3d(self, matrix):
        """Reduce high-dim matrix to 3D using best available method."""
        import numpy as np

        # Try UMAP first
        try:
            from umap import UMAP
            reducer = UMAP(
                n_components=3,
                metric="cosine",
                n_neighbors=min(15, len(matrix) - 1),
                min_dist=0.1,
                random_state=self.random_state,
            )
            return reducer.fit_transform(matrix)
        except ImportError:
            pass

        # Fall back to PCA
        try:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3, random_state=self.random_state)
            return reducer.fit_transform(matrix)
        except ImportError:
            pass

        # Last resort: random projection
        rng = np.random.RandomState(self.random_state)
        proj = rng.randn(matrix.shape[1], 3)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        return matrix @ proj

    def _normalize(self, coords, scale: float = 200.0):
        """Normalize coordinates to [-scale, scale] range per axis."""
        import numpy as np
        for axis in range(3):
            col = coords[:, axis]
            mn, mx = col.min(), col.max()
            if mx - mn > 1e-10:
                coords[:, axis] = (col - mn) / (mx - mn) * 2 * scale - scale
            else:
                coords[:, axis] = 0.0
        return coords


# ═══════════════════════════════════════════════════════════
# Star Map Renderer
# ═══════════════════════════════════════════════════════════


class StarMapRenderer:
    """Generate interactive 3D star map HTML."""

    def render(
        self,
        engine: "STGEngine",
        config: VisualizationConfig,
        coordinates: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> str:
        """Generate complete HTML string for the 3D star map."""
        node_names, edge_pairs = self._extract_subgraph(engine, config)
        activation_map = self._get_activation_map(engine, config)

        # Compute degree map
        degree: Dict[str, int] = defaultdict(int)
        for src, tgt in edge_pairs:
            degree[src] += 1
            degree[tgt] += 1

        max_degree = max(degree.values()) if degree else 1

        # Build node data
        nodes_data = []
        for name in node_names:
            node = engine._nodes.get(engine._nk(name))
            ns = node.namespace if node else None
            color = NAMESPACE_COLORS.get(ns, DEFAULT_NODE_COLOR) if ns else DEFAULT_NODE_COLOR

            # Size based on degree or activation
            if activation_map and name in activation_map:
                size_factor = activation_map[name]
            else:
                size_factor = degree.get(name, 1) / max(max_degree, 1)

            size = config.node_size_min + (config.node_size_max - config.node_size_min) * min(size_factor, 1.0)

            nd: Dict[str, Any] = {
                "id": name,
                "name": name,
                "color": color,
                "size": round(size, 2),
                "namespace": ns or "",
            }

            if node:
                nd["type"] = node.anchor_type or ""
                nd["tension"] = round(node.tension, 4)
                nd["activation"] = round(node.activation, 4)

            if coordinates and name in coordinates:
                x, y, z = coordinates[name]
                nd["fx"] = round(x, 2)
                nd["fy"] = round(y, 2)
                nd["fz"] = round(z, 2)

            nodes_data.append(nd)

        # Build edge data
        edges_data = []
        for src, tgt in edge_pairs:
            edge = engine._edges_lookup.get((src, tgt))
            if edge:
                rule = edge.rule
                color = RULE_COLORS.get(rule, DEFAULT_EDGE_COLOR) if rule else DEFAULT_EDGE_COLOR
                width = 0.3 + 1.7 * edge.confidence * edge.salience
                edge_class = edge.modifiers.get("edge_class", "")
            else:
                color = DEFAULT_EDGE_COLOR
                width = 0.5
                edge_class = ""

            ed: Dict[str, Any] = {
                "source": src,
                "target": tgt,
                "color": color,
                "width": round(width, 2),
            }
            if edge:
                ed["confidence"] = round(edge.confidence, 3)
                ed["salience"] = round(edge.salience, 3)
                if edge.rule:
                    ed["rule"] = edge.rule
                if edge_class:
                    ed["edgeClass"] = edge_class
                # Include key modifiers (skip internal ones)
                display_mods = {
                    k: v for k, v in edge.modifiers.items()
                    if not k.startswith("_") and k != "edge_class"
                }
                if display_mods:
                    ed["modifiers"] = display_mods
            edges_data.append(ed)

        graph_data = {"nodes": nodes_data, "links": edges_data}

        # Build namespace legend
        ns_counts: Dict[str, int] = defaultdict(int)
        for nd in nodes_data:
            ns = nd.get("namespace", "")
            if ns:
                ns_counts[ns] += 1

        return self._generate_html(graph_data, config, ns_counts)

    def _extract_subgraph(
        self,
        engine: "STGEngine",
        config: VisualizationConfig,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract node and edge lists based on mode."""
        if config.mode == "ego" and config.ego_node:
            return self._extract_ego(engine, config.ego_node, config.ego_depth)
        elif config.mode == "community" and config.community_id is not None:
            return self._extract_community(engine, config.community_id)
        elif config.mode == "propagate" and config.propagate_query:
            return self._extract_propagate(engine, config.propagate_query)
        else:
            return self._extract_full(engine, config.filter_dead_ends)

    def _extract_full(
        self,
        engine: "STGEngine",
        filter_dead_ends: bool = False,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Full graph. Optionally filter degree-1 nodes."""
        # Use display names so node ids match edge endpoints (STGEdge stores
        # source/target as display names; engine._nodes keys are normalized).
        all_nodes = {node.name for node in engine._nodes.values()}
        all_edges = [(e.source, e.target) for e in engine._edges]

        if filter_dead_ends:
            degree: Dict[str, int] = defaultdict(int)
            for src, tgt in all_edges:
                degree[src] += 1
                degree[tgt] += 1
            all_nodes = {n for n in all_nodes if degree.get(n, 0) > 1}
            all_edges = [(s, t) for s, t in all_edges if s in all_nodes and t in all_nodes]

        return list(all_nodes), all_edges

    def _extract_ego(
        self,
        engine: "STGEngine",
        center: str,
        depth: int,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """BFS ego network."""
        visited: Set[str] = {center}
        frontier: Set[str] = {center}

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                for edge in engine._edges:
                    if edge.source == node and edge.target not in visited:
                        next_frontier.add(edge.target)
                    if edge.target == node and edge.source not in visited:
                        next_frontier.add(edge.source)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        edges = [
            (e.source, e.target) for e in engine._edges
            if e.source in visited and e.target in visited
        ]
        return list(visited), edges

    def _extract_community(
        self,
        engine: "STGEngine",
        community_id: int,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Nodes in a Louvain community."""
        try:
            from stg_engine.topology import TopologyAnalyzer
            analyzer = TopologyAnalyzer()
            communities = analyzer.detect_communities(engine)
            if community_id < len(communities):
                members = set(communities[community_id]["members"])
            else:
                members = set()
        except Exception:
            members = set()

        edges = [
            (e.source, e.target) for e in engine._edges
            if e.source in members and e.target in members
        ]
        return list(members), edges

    def _extract_propagate(
        self,
        engine: "STGEngine",
        query: str,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Activated nodes from propagation."""
        activated_names = engine.propagate(query)
        activated = set(activated_names)

        edges = [
            (e.source, e.target) for e in engine._edges
            if e.source in activated and e.target in activated
        ]
        return list(activated), edges

    def _get_activation_map(
        self,
        engine: "STGEngine",
        config: VisualizationConfig,
    ) -> Optional[Dict[str, float]]:
        """Get activation values if in propagate mode."""
        if config.mode == "propagate" and config.propagate_query:
            activated_names = engine.propagate(config.propagate_query)
            if not activated_names:
                return None
            # propagate() returns display names; engine._nodes is keyed by
            # normalized form, so look up via _nk.
            activation: Dict[str, float] = {}
            for n in activated_names:
                node = engine._nodes.get(engine._nk(n))
                if node is not None:
                    activation[n] = node.activation
            max_a = max(activation.values()) if activation else 1.0
            if max_a > 0:
                return {n: a / max_a for n, a in activation.items()}
        return None

    def _generate_html(
        self,
        graph_data: Dict,
        config: VisualizationConfig,
        ns_counts: Dict[str, int],
    ) -> str:
        """Generate the complete HTML file."""
        data_json = json.dumps(graph_data, ensure_ascii=False)
        n_nodes = len(graph_data["nodes"])
        n_edges = len(graph_data["links"])

        # Build namespace legend HTML
        legend_items = []
        for ns, count in sorted(ns_counts.items(), key=lambda x: -x[1]):
            color = NAMESPACE_COLORS.get(ns, DEFAULT_NODE_COLOR)
            legend_items.append(
                f'<div class="legend-item">'
                f'<span class="dot" style="background:{color}"></span>'
                f'{ns} ({count})</div>'
            )
        legend_html = "\n".join(legend_items)

        # Rule legend
        rule_legend = []
        for rule, color in RULE_COLORS.items():
            rule_legend.append(
                f'<div class="legend-item">'
                f'<span class="line" style="background:{color}"></span>'
                f'{rule}</div>'
            )
        rule_legend_html = "\n".join(rule_legend)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>STG Star Map — {n_nodes} nodes, {n_edges} edges</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: {config.background_color}; overflow: hidden; font-family: 'Segoe UI', system-ui, sans-serif; }}
  #graph {{ width: 100vw; height: 100vh; }}

  #info-panel {{
    position: fixed; top: 12px; left: 12px;
    background: rgba(0,0,20,0.85); color: #ccc;
    padding: 12px 16px; border-radius: 8px;
    font-size: 13px; min-width: 180px;
    border: 1px solid rgba(100,100,255,0.2);
    backdrop-filter: blur(8px);
    z-index: 10;
  }}
  #info-panel h3 {{ color: #fff; margin-bottom: 8px; font-size: 15px; }}
  .stat {{ margin: 2px 0; }}
  .stat span {{ color: #7fb3ff; }}

  #search-box {{
    position: fixed; top: 12px; right: 12px;
    z-index: 10;
  }}
  #search-box input {{
    background: rgba(0,0,20,0.85); color: #fff;
    border: 1px solid rgba(100,100,255,0.3);
    padding: 8px 14px; border-radius: 20px;
    font-size: 14px; width: 240px;
    outline: none;
    backdrop-filter: blur(8px);
  }}
  #search-box input::placeholder {{ color: #667; }}
  #search-box input:focus {{ border-color: rgba(100,100,255,0.6); }}

  #legend {{
    position: fixed; bottom: 12px; left: 12px;
    background: rgba(0,0,20,0.85); color: #ccc;
    padding: 10px 14px; border-radius: 8px;
    font-size: 12px; max-height: 40vh; overflow-y: auto;
    border: 1px solid rgba(100,100,255,0.2);
    backdrop-filter: blur(8px);
    z-index: 10;
  }}
  #legend h4 {{ color: #fff; margin-bottom: 6px; font-size: 13px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
  .line {{ width: 16px; height: 3px; display: inline-block; flex-shrink: 0; border-radius: 2px; }}

  #tooltip {{
    position: fixed; display: none;
    background: rgba(0,0,30,0.92); color: #eee;
    padding: 10px 14px; border-radius: 8px;
    font-size: 12px; pointer-events: none;
    border: 1px solid rgba(100,100,255,0.3);
    max-width: 320px; z-index: 20;
    backdrop-filter: blur(8px);
  }}
  #tooltip .tt-name {{ color: #7fb3ff; font-weight: bold; font-size: 14px; }}
  #tooltip .tt-ns {{ color: #aaa; font-size: 11px; }}
  #tooltip .tt-row {{ margin: 2px 0; }}

  #mode-label {{
    position: fixed; top: 12px; left: 50%;
    transform: translateX(-50%);
    color: rgba(255,255,255,0.3); font-size: 12px;
    z-index: 10;
  }}
</style>
</head>
<body>

<div id="info-panel">
  <h3>STG Star Map</h3>
  <div class="stat">Nodes: <span>{n_nodes}</span></div>
  <div class="stat">Edges: <span>{n_edges}</span></div>
  <div class="stat">Mode: <span>{config.mode}</span></div>
</div>

<div id="search-box">
  <input type="text" id="search" placeholder="Search nodes..." autocomplete="off">
</div>

<div id="legend">
  <h4>Namespaces</h4>
  {legend_html}
  <h4 style="margin-top:8px">Edge Rules</h4>
  {rule_legend_html}
</div>

<div id="tooltip"></div>
<div id="mode-label">{config.mode} mode</div>
<div id="graph"></div>

<script src="https://unpkg.com/3d-force-graph@1"></script>
<script>
const graphData = {data_json};
const edgeOpacity = {config.edge_opacity};
const labelThreshold = {config.label_threshold};

let highlightNode = null;
let highlightLinks = new Set();
let highlightNeighbors = new Set();
let searchTerm = '';

// Build adjacency for highlight
const nodeById = {{}};
graphData.nodes.forEach(n => {{ nodeById[n.id] = n; }});

const graph = ForceGraph3D()(document.getElementById('graph'))
  .graphData(graphData)
  .backgroundColor('{config.background_color}')
  .nodeLabel('')
  .nodeVal(n => n.size)
  .nodeColor(n => {{
    if (searchTerm && !n.id.toLowerCase().includes(searchTerm)) return 'rgba(60,60,80,0.3)';
    if (highlightNode) {{
      if (n.id === highlightNode) return '#ffffff';
      if (highlightNeighbors.has(n.id)) return n.color;
      return 'rgba(60,60,80,0.3)';
    }}
    return n.color;
  }})
  .nodeOpacity(0.9)
  .linkColor(l => {{
    if (highlightNode && highlightLinks.has(l)) return l.color || '#888';
    return l.color || '{DEFAULT_EDGE_COLOR}';
  }})
  .linkWidth(l => {{
    if (highlightNode && highlightLinks.has(l)) return (l.width || 0.5) * 2;
    return l.width || 0.5;
  }})
  .linkOpacity(l => {{
    if (highlightNode) {{
      return highlightLinks.has(l) ? 0.8 : 0.03;
    }}
    return edgeOpacity;
  }})
  .linkDirectionalArrowLength(2)
  .linkDirectionalArrowRelPos(1)
  .linkLabel('')
  .onLinkHover(link => {{
    const tooltip = document.getElementById('tooltip');
    if (link) {{
      const sid = typeof link.source === 'object' ? link.source.id : link.source;
      const tid = typeof link.target === 'object' ? link.target.id : link.target;
      tooltip.style.display = 'block';
      let html = '<div class="tt-name">[' + sid + '] &rarr; [' + tid + ']</div>';
      if (link.rule) html += '<div class="tt-row">Rule: <span style="color:#7fb3ff">' + link.rule + '</span></div>';
      if (link.confidence !== undefined) html += '<div class="tt-row">Confidence: ' + link.confidence + '</div>';
      if (link.salience !== undefined) html += '<div class="tt-row">Salience: ' + link.salience + '</div>';
      if (link.edgeClass) html += '<div class="tt-row">Class: ' + link.edgeClass + '</div>';
      if (link.modifiers) {{
        const mods = link.modifiers;
        const keys = Object.keys(mods).slice(0, 6);
        if (keys.length > 0) {{
          html += '<div style="margin-top:4px;border-top:1px solid rgba(100,100,255,0.2);padding-top:4px">';
          keys.forEach(k => {{
            let v = mods[k];
            if (typeof v === 'string' && v.length > 60) v = v.substring(0, 57) + '...';
            html += '<div class="tt-row" style="font-size:11px"><span style="color:#888">' + k + ':</span> ' + v + '</div>';
          }});
          html += '</div>';
        }}
      }}
      tooltip.innerHTML = html;
      document.body.style.cursor = 'pointer';
    }} else {{
      tooltip.style.display = 'none';
      document.body.style.cursor = 'default';
    }}
  }})
  .onNodeHover(node => {{
    const tooltip = document.getElementById('tooltip');
    if (node) {{
      tooltip.style.display = 'block';
      let html = '<div class="tt-name">' + node.id + '</div>';
      if (node.namespace) html += '<div class="tt-ns">' + node.namespace + '</div>';
      if (node.type) html += '<div class="tt-row">Type: ' + node.type + '</div>';
      if (node.tension !== undefined) html += '<div class="tt-row">Tension: ' + node.tension + '</div>';
      if (node.activation !== undefined) html += '<div class="tt-row">Activation: ' + node.activation + '</div>';
      tooltip.innerHTML = html;
      document.body.style.cursor = 'pointer';
    }} else {{
      tooltip.style.display = 'none';
      document.body.style.cursor = 'default';
    }}
  }})
  .onNodeClick(node => {{
    if (highlightNode === node.id) {{
      // Deselect
      highlightNode = null;
      highlightLinks.clear();
      highlightNeighbors.clear();
    }} else {{
      highlightNode = node.id;
      highlightLinks.clear();
      highlightNeighbors.clear();
      graphData.links.forEach(l => {{
        const sid = typeof l.source === 'object' ? l.source.id : l.source;
        const tid = typeof l.target === 'object' ? l.target.id : l.target;
        if (sid === node.id || tid === node.id) {{
          highlightLinks.add(l);
          highlightNeighbors.add(sid);
          highlightNeighbors.add(tid);
        }}
      }});
    }}
    graph.nodeColor(graph.nodeColor());
    graph.linkOpacity(graph.linkOpacity());
    graph.linkWidth(graph.linkWidth());
    graph.linkColor(graph.linkColor());
  }})
  .onBackgroundClick(() => {{
    highlightNode = null;
    highlightLinks.clear();
    highlightNeighbors.clear();
    graph.nodeColor(graph.nodeColor());
    graph.linkOpacity(graph.linkOpacity());
    graph.linkWidth(graph.linkWidth());
    graph.linkColor(graph.linkColor());
  }})
  .onNodeDragEnd(node => {{
    node.fx = node.x;
    node.fy = node.y;
    node.fz = node.z;
  }});

// Tooltip follow mouse
document.addEventListener('mousemove', e => {{
  const tooltip = document.getElementById('tooltip');
  if (tooltip.style.display === 'block') {{
    tooltip.style.left = (e.clientX + 15) + 'px';
    tooltip.style.top = (e.clientY + 15) + 'px';
  }}
}});

// Search
document.getElementById('search').addEventListener('input', e => {{
  searchTerm = e.target.value.toLowerCase();
  graph.nodeColor(graph.nodeColor());
}});

// Keyboard shortcut: Escape to clear
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    document.getElementById('search').value = '';
    searchTerm = '';
    highlightNode = null;
    highlightLinks.clear();
    highlightNeighbors.clear();
    graph.nodeColor(graph.nodeColor());
    graph.linkOpacity(graph.linkOpacity());
    graph.linkWidth(graph.linkWidth());
    graph.linkColor(graph.linkColor());
  }}
}});
</script>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════


def generate_starmap(
    engine: "STGEngine",
    mode: str = "full",
    output: Optional[str] = None,
    auto_open: bool = True,
    use_embeddings: bool = True,
    **kwargs: Any,
) -> str:
    """Generate and optionally open a 3D star map visualization.

    Args:
        engine: STGEngine instance with loaded graph.
        mode: "full" | "ego" | "community" | "propagate"
        output: Output file path (default: starmap.html in CWD)
        auto_open: Open browser automatically
        use_embeddings: Use UMAP/PCA coordinates from embeddings
        **kwargs: Mode-specific options:
            ego_node (str), ego_depth (int),
            community_id (int), query (str),
            filter_dead_ends (bool)

    Returns:
        Path to generated HTML file.
    """
    config = VisualizationConfig(
        mode=mode,
        ego_node=kwargs.get("ego_node"),
        ego_depth=kwargs.get("ego_depth", 2),
        community_id=kwargs.get("community_id"),
        propagate_query=kwargs.get("query"),
        use_embeddings=use_embeddings,
        output_path=output,
        auto_open=auto_open,
        filter_dead_ends=kwargs.get("filter_dead_ends", False),
    )

    # Compute coordinates from embeddings if available
    coordinates: Optional[Dict[str, Tuple[float, float, float]]] = None
    if use_embeddings and engine._vector_index is not None and engine._vector_index.size > 0:
        coord_engine = CoordinateEngine(random_state=config.random_state)
        coordinates = coord_engine.compute_coordinates(engine)

    renderer = StarMapRenderer()
    html = renderer.render(engine, config, coordinates)

    out_path = output or "starmap.html"
    Path(out_path).write_text(html, encoding="utf-8")

    if auto_open:
        webbrowser.open(str(Path(out_path).resolve()))

    return str(Path(out_path).resolve())
