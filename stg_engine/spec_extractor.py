"""STLC specification extractor for STG.

Extracts structured information from STLC specification files using
regex patterns matched to the consistent STLC format:
- Header metadata (blockquote key-value pairs)
- Section hierarchy (## N.N Title)
- Class definitions ([Entry_X] -> [X_Definition])
- Method signatures ([Entry_X_Method] -> [X_Signature])

Usage:
    from stg_engine.spec_extractor import extract_spec
    count = extract_spec("path/to/SPEC.md", engine, namespace="Spec")
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from stg_engine.concept_skeleton import CORE_CONCEPT_NAMES

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ── Regex Patterns ──────────────────────────────────────────────────

# Blockquote header metadata: > **Key:** Value
HEADER_PATTERN = re.compile(r'>\s*\*\*(\w[\w\s]*?)\s*(?:：|:)\s*\*\*\s*(.*)')

# Section headings: ## 1. Title  or  ### 1.1. Title
SECTION_PATTERN = re.compile(r'^(#{2,3})\s+(\d+(?:\.\d+)*)\.?\s+(.*)')

# STL class definition: [Entry_ClassName] -> [ClassName_Definition]
CLASS_PATTERN = re.compile(
    r'\[Entry_(\w+)\]\s*(?:->|→)\s*\[(\w+)_Definition\]'
)

# STL method definition: [Entry_MethodName_Method] -> [MethodName_Signature]
METHOD_PATTERN = re.compile(
    r'\[Entry_(\w+?)_Method\]\s*(?:->|→)\s*\[(\w+?)_Signature\]'
)

# Signature in modifier: signature="..."
SIGNATURE_PATTERN = re.compile(r'signature="([^"]*)"')

# Phase reference: phase1, phase2, Phase 4, etc.
PHASE_PATTERN = re.compile(r'[Pp]hase[\s_]?(\d+)', re.IGNORECASE)

# STL code block: ```stl ... ``` (fenced) or standalone STL statements
STL_BLOCK_PATTERN = re.compile(
    r'```stl\s*\n(.*?)```',
    re.DOTALL,
)

# Standalone STL statement (not in code block): [Anchor] -> [Anchor] ::mod(...)
STL_STATEMENT_PATTERN = re.compile(
    r'^\s*(\[[\w:_-]+\]\s*(?:->|→)\s*\[[\w:_-]+\]\s*::mod\(.*?\))\s*$',
    re.MULTILINE | re.DOTALL,
)


@dataclass
class _SpecMeta:
    """Extracted header metadata from a spec file."""
    fields: Dict[str, str]


def _sanitize_name(text: str) -> str:
    """Convert text into a valid STL anchor name."""
    text = re.sub(r'[*`#\[\](){}!|>]', '', text)
    text = re.sub(r'[\s\-]+', '_', text.strip())
    text = re.sub(r'[^\w:]', '', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or "Unnamed"


def _derive_spec_name(filepath: str) -> str:
    """Derive a spec node name from file path."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return _sanitize_name(basename)


def _extract_header(lines: List[str]) -> _SpecMeta:
    """Extract blockquote metadata from file header."""
    fields: Dict[str, str] = {}
    for line in lines[:30]:  # Headers are always in first 30 lines
        m = HEADER_PATTERN.match(line)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            fields[key] = value
    return _SpecMeta(fields=fields)


def _extract_phase_refs(text: str) -> List[str]:
    """Find phase references in text, return matching Phase_N node names."""
    phases = set()
    for m in PHASE_PATTERN.finditer(text):
        phase_num = m.group(1)
        phase_name = f"Phase_{phase_num}"
        if phase_name in CORE_CONCEPT_NAMES:
            phases.add(phase_name)
    return list(phases)


def _extract_embedded_stl(
    content: str,
    engine: "STGEngine",
    session_id: Optional[str] = None,
) -> int:
    """Extract and ingest all embedded STL statements from markdown content.

    Finds STL in:
    1. Fenced ```stl code blocks
    2. Standalone [A] -> [B] ::mod(...) lines outside code blocks

    Returns:
        Number of edges ingested.
    """
    stl_texts: List[str] = []

    # 1. Extract from ```stl blocks
    for m in STL_BLOCK_PATTERN.finditer(content):
        block = m.group(1).strip()
        if block:
            stl_texts.append(block)

    # 2. Remove all fenced code blocks, then find standalone STL statements
    content_no_blocks = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    for m in STL_STATEMENT_PATTERN.finditer(content_no_blocks):
        stmt = m.group(1).strip()
        if stmt:
            stl_texts.append(stmt)

    if not stl_texts:
        return 0

    # Ingest each block independently — some may fail (e.g. parser
    # limitations with Chinese modifier values) but others succeed.
    count = 0
    for text in stl_texts:
        try:
            count += engine.ingest_stl(text, session_id=session_id, auto_virtual=False)
        except Exception:
            continue
    return count


def extract_spec(
    filepath: str,
    engine: "STGEngine",
    namespace: str = "Spec",
    project_root: Optional[str] = None,
) -> int:
    """Extract structure from an STLC specification into the STG.

    Creates:
    - 1 spec document node (with metadata from header)
    - N section nodes (with line ranges and summaries)
    - M class nodes (from [Entry_X] -> [X_Definition] patterns)
    - K method nodes (from [Entry_X_Method] -> [X_Signature] patterns)
    - Containment, definition, and bridge edges

    Args:
        filepath: Path to the STLC spec file
        engine: STGEngine to add nodes/edges into
        namespace: Namespace prefix for created nodes
        project_root: Base directory for relative path storage

    Returns:
        Count of elements added
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    lines = content.split('\n')

    count = 0
    spec_name = _derive_spec_name(filepath)
    spec_node_name = f"{namespace}:{spec_name}"

    rel_path = filepath
    if project_root:
        try:
            rel_path = os.path.relpath(filepath, project_root)
        except ValueError:
            pass

    # 1. Extract header metadata
    meta = _extract_header(lines)

    # Extract title from first # heading
    title = spec_name
    for line in lines[:20]:
        m = re.match(r'^#\s+(.+)', line)
        if m:
            title = m.group(1).strip()
            break

    # Create spec document node
    node_meta = {
        "file": rel_path,
        "lines": f"1-{len(lines)}",
        "title": title,
    }
    # Add header fields as metadata
    for key, value in meta.fields.items():
        safe_key = _sanitize_name(key).lower()
        node_meta[safe_key] = value

    engine.add_node(
        spec_node_name,
        namespace=namespace,
        anchor_type="Entity",
        **node_meta,
    )
    count += 1

    # 2. Extract sections (## N. Title)
    sections = []
    for i, line in enumerate(lines):
        m = SECTION_PATTERN.match(line)
        if m:
            level = len(m.group(1))
            number = m.group(2)
            sec_title = m.group(3).strip()
            sections.append({
                "number": number,
                "title": sec_title,
                "level": level,
                "line_start": i + 1,
            })

    # Compute line_end
    for i, sec in enumerate(sections):
        if i + 1 < len(sections):
            sec["line_end"] = sections[i + 1]["line_start"] - 1
        else:
            sec["line_end"] = len(lines)

    # Extract summary for each section
    for sec in sections:
        for j in range(sec["line_start"], min(sec["line_end"], len(lines))):
            line = lines[j].strip()
            if line and not line.startswith('#') and not line.startswith('>'):
                sec["summary"] = line[:100]
                break
        else:
            sec["summary"] = ""

    # Create section nodes
    for sec in sections:
        sec_safe = _sanitize_name(sec["title"])
        if not sec_safe:
            continue
        sec_node_name = f"{namespace}:{spec_name}:{sec['number']}_{sec_safe}"

        engine.add_node(
            sec_node_name,
            namespace=namespace,
            anchor_type="PathSegment",
            file=rel_path,
            lines=f"{sec['line_start']}-{sec['line_end']}",
            summary=sec["summary"],
            section_number=sec["number"],
        )
        count += 1

        engine.add_edge(
            spec_node_name, sec_node_name,
            confidence=1.0,
            strength=0.8,
            rule="definitional",
            type="contains",
            edge_class="structural",
        )
        count += 1

    # 3. Extract class definitions
    for m in CLASS_PATTERN.finditer(content):
        class_name = m.group(1)
        class_node_name = f"{namespace}:{spec_name}:Class_{class_name}"

        engine.add_node(
            class_node_name,
            namespace=namespace,
            anchor_type="Entity",
            element_type="class",
            spec=spec_name,
        )
        count += 1

        engine.add_edge(
            spec_node_name, class_node_name,
            confidence=0.95,
            strength=0.85,
            rule="definitional",
            type="defines",
            edge_class="structural",
        )
        count += 1

    # 4. Extract method definitions
    for m in METHOD_PATTERN.finditer(content):
        method_name = m.group(1)
        method_node_name = f"{namespace}:{spec_name}:Method_{method_name}"

        # Try to find signature nearby
        start_pos = m.start()
        context = content[start_pos:start_pos + 500]
        sig_match = SIGNATURE_PATTERN.search(context)
        signature = sig_match.group(1) if sig_match else None

        node_kwargs = {
            "element_type": "method",
            "spec": spec_name,
        }
        if signature:
            node_kwargs["signature"] = signature

        engine.add_node(
            method_node_name,
            namespace=namespace,
            anchor_type="PathSegment",
            **node_kwargs,
        )
        count += 1

        engine.add_edge(
            spec_node_name, method_node_name,
            confidence=0.95,
            strength=0.85,
            rule="definitional",
            type="defines",
            edge_class="structural",
        )
        count += 1

    # 5. Bridge to core concepts
    for concept_name in CORE_CONCEPT_NAMES:
        # Use simple substring check for performance
        pattern = concept_name.replace('_', r'[\s_]')
        if re.search(r'\b' + pattern + r'\b', content, re.IGNORECASE):
            engine.add_edge(
                spec_node_name, concept_name,
                confidence=0.80,
                strength=0.6,
                rule="logical",
                type="references",
                edge_class="structural",
            )
            count += 1

    # 6. Bridge to phases
    for phase_name in _extract_phase_refs(content):
        engine.add_edge(
            spec_node_name, phase_name,
            confidence=0.90,
            strength=0.7,
            rule="definitional",
            type="phase_of",
            edge_class="structural",
        )
        count += 1

    # 7. Bridge to STLC_Specification concept
    if "STLC_Specification" in CORE_CONCEPT_NAMES:
        engine.add_edge(
            spec_node_name, "STLC_Specification",
            confidence=0.95,
            strength=0.8,
            rule="definitional",
            type="instance_of",
            edge_class="structural",
        )
        count += 1

    # 8. Extract embedded STL statements (with full modifiers)
    stl_count = _extract_embedded_stl(content, engine)
    count += stl_count

    return count
