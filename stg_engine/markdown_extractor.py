"""Generic markdown structure extractor for STG.

Extracts document structure (title, sections) from arbitrary markdown files
and creates skeletal nodes with file location + summary metadata. Also bridges
to core concepts via keyword matching.

Usage:
    from stg_engine.markdown_extractor import extract_markdown_structure
    count = extract_markdown_structure("path/to/doc.md", engine, namespace="Doc")
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from stg_engine.concept_skeleton import CORE_CONCEPT_NAMES

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


@dataclass
class _Section:
    """Internal representation of a markdown section."""
    title: str
    level: int
    line_start: int
    line_end: int = 0
    summary: str = ""


def _sanitize_name(text: str) -> str:
    """Convert a heading into a valid STL anchor name.

    Rules: PascalCase-ish, underscores for separation, strip special chars.
    """
    # Remove markdown formatting
    text = re.sub(r'[*`#\[\](){}!|>]', '', text)
    # Replace spaces/hyphens with underscores
    text = re.sub(r'[\s\-]+', '_', text.strip())
    # Remove remaining invalid chars (keep alphanumeric, underscore, colon)
    text = re.sub(r'[^\w:]', '', text)
    # Collapse multiple underscores
    text = re.sub(r'_+', '_', text)
    # Strip leading/trailing underscores
    text = text.strip('_')
    return text or "Unnamed"


def _derive_doc_name(filepath: str) -> str:
    """Derive a document node name from file path.

    Examples:
        'development/specifications/MEMORY_ARCHITECTURE.md' → 'MEMORY_ARCHITECTURE'
        'foundation/CONSCIOUSNESS_FOUNDATIONS.md' → 'CONSCIOUSNESS_FOUNDATIONS'
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return _sanitize_name(basename)


def _find_sections(lines: List[str]) -> List[_Section]:
    """Extract sections from markdown lines (## and ### headings)."""
    sections: List[_Section] = []
    heading_re = re.compile(r'^(#{2,3})\s+(.+)')

    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            sections.append(_Section(
                title=title,
                level=level,
                line_start=i + 1,  # 1-based
            ))

    # Compute line_end for each section (next section start - 1, or EOF)
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section.line_end = sections[i + 1].line_start - 1
        else:
            section.line_end = len(lines)

    # Extract summary: first non-empty line after heading
    for section in sections:
        for j in range(section.line_start, min(section.line_end, len(lines))):
            line = lines[j].strip()
            if line and not line.startswith('#'):
                section.summary = line[:100]
                break

    return sections


def _concept_mentioned_in(text: str, concept_name: str) -> bool:
    """Check if a core concept is mentioned in text (case-insensitive).

    Uses word boundary matching to avoid false positives
    (e.g. 'Phase_1' shouldn't match 'Phase_10').
    """
    # Convert underscores to flexible whitespace/underscore pattern
    pattern_str = concept_name.replace('_', r'[\s_]')
    pattern = re.compile(r'\b' + pattern_str + r'\b', re.IGNORECASE)
    return bool(pattern.search(text))


def extract_markdown_structure(
    filepath: str,
    engine: "STGEngine",
    namespace: str = "Doc",
    project_root: Optional[str] = None,
) -> int:
    """Extract structure from a markdown file into the STG.

    Creates:
    - 1 document node (with file path, total lines, title)
    - N section nodes (with file path, line range, summary)
    - N containment edges (doc → section)
    - M bridge edges to core concepts (by keyword matching)

    Args:
        filepath: Absolute or relative path to the markdown file
        engine: STGEngine to add nodes/edges into
        namespace: Namespace prefix for created nodes
        project_root: Base directory for relative path storage

    Returns:
        Count of elements added (nodes + edges)
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    count = 0
    doc_name = _derive_doc_name(filepath)
    doc_node_name = f"{namespace}:{doc_name}"

    # Store relative path if project_root provided
    rel_path = filepath
    if project_root:
        try:
            rel_path = os.path.relpath(filepath, project_root)
        except ValueError:
            pass  # Different drives on Windows

    # 1. Create document node
    # Extract title from first # heading
    title = doc_name
    for line in lines[:20]:  # Check first 20 lines
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

    # 2. Extract and create section nodes
    sections = _find_sections(lines)
    for section in sections:
        section_name_part = _sanitize_name(section.title)
        if not section_name_part:
            continue
        section_node_name = f"{namespace}:{doc_name}:{section_name_part}"

        engine.add_node(
            section_node_name,
            namespace=namespace,
            anchor_type="PathSegment",
            file=rel_path,
            lines=f"{section.line_start}-{section.line_end}",
            summary=section.summary,
        )
        count += 1

        # Containment edge: doc → section
        engine.add_edge(
            doc_node_name, section_node_name,
            confidence=1.0,
            strength=0.7,
            rule="definitional",
            type="contains",
            edge_class="structural",
        )
        count += 1

    # 3. Bridge to core concepts
    full_text = ''.join(lines)
    for concept_name in CORE_CONCEPT_NAMES:
        if _concept_mentioned_in(full_text, concept_name):
            engine.add_edge(
                doc_node_name, concept_name,
                confidence=0.75,
                strength=0.4,
                rule="logical",
                type="references",
                edge_class="structural",
            )
            count += 1

    return count
