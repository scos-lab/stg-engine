"""MemoryMatrix importer for STG Engine.

Parses memoryMatrix.md (Syn-claude's episodic memory log) and populates
the STG Engine with sessions, events, tensions, belief evolutions,
and STL edges.

The memoryMatrix has multiple sections:
  0. MEMORY INDEX — temporal, semantic, importance, tension, belief evolution indexes
  1. SESSION RECORDS — chronological session logs with STL statements
  2. EVENT blocks — structured event metadata (E001-E048+)
  3. Closure events — YAML-like session summary blocks
"""

import re
from typing import Dict, List, Optional, Tuple

from stg_engine.engine import STGEngine
from stg_engine.types import (
    STGSession, STGEvent, STGTension, STGBeliefEvolution,
)


def import_memory_matrix(path: str, engine: Optional[STGEngine] = None) -> STGEngine:
    """Import a memoryMatrix.md file into an STG Engine.

    Args:
        path: Path to memoryMatrix.md
        engine: Optional existing engine to populate. Creates new if None.

    Returns:
        Populated STGEngine
    """
    if engine is None:
        engine = STGEngine()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    # Parse in phases
    _parse_temporal_index(lines, engine)
    _parse_tension_index(lines, engine)
    _parse_belief_evolution_index(lines, engine)
    _parse_events(lines, engine)
    _ingest_all_stl(content, engine)

    return engine


def _find_section(lines: List[str], header: str) -> Tuple[int, int]:
    """Find start and end line of a section by header text.

    Returns (start, end) where start is the header line index
    and end is the next section header or EOF.
    """
    start = -1
    for i, line in enumerate(lines):
        if header in line:
            start = i
            break

    if start == -1:
        return -1, -1

    # Find end (next ## or ### at same or higher level, or ----)
    header_level = len(line.split()[0]) if line.strip().startswith("#") else 0
    end = len(lines)

    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            # Count level
            level = 0
            for ch in stripped:
                if ch == "#":
                    level += 1
                else:
                    break
            if level <= header_level and header_level > 0:
                end = i
                break

    return start, end


def _parse_temporal_index(lines: List[str], engine: STGEngine) -> None:
    """Parse the TEMPORAL INDEX section to extract session metadata."""
    start, end = _find_section(lines, "TEMPORAL INDEX")
    if start == -1:
        return

    # Pattern: - 2026-02-08: SESSION_020 → [E045-E047] (I_avg=0.87) ⭐ TITLE
    pattern = re.compile(
        r'-\s*(\d{4}-\d{2}-\d{2}):\s*'       # date
        r'(SESSION_\d+(?:-\d+)?)\s*'           # session_id (may include ranges like 013-014)
        r'(?:→\s*\[([^\]]*)\])?\s*'            # optional event range [E045-E047]
        r'(?:\(I_avg=([\d.]+)\))?\s*'          # optional importance avg
        r'(?:⭐\s*)?'                           # optional star
        r'(.*)',                                # title
    )

    for i in range(start + 1, end):
        match = pattern.search(lines[i])
        if not match:
            continue

        date = match.group(1)
        session_id_raw = match.group(2)
        importance_str = match.group(4)
        title = match.group(5).strip() if match.group(5) else ""

        avg_importance = float(importance_str) if importance_str else 0.0

        # Handle ranges like SESSION_013-014
        if "-" in session_id_raw and not session_id_raw.startswith("SESSION_0"):
            # e.g., SESSION_013-014
            parts = session_id_raw.split("-")
            base = parts[0]  # SESSION_013
            suffix = parts[1]  # 014
            prefix = base.rsplit("_", 1)[0]  # SESSION
            for sid in [base, f"{prefix}_{suffix}"]:
                engine.add_session(STGSession(
                    session_id=sid,
                    date=date,
                    title=title,
                    avg_importance=avg_importance,
                ))
        else:
            engine.add_session(STGSession(
                session_id=session_id_raw,
                date=date,
                title=title,
                avg_importance=avg_importance,
            ))


def _parse_tension_index(lines: List[str], engine: STGEngine) -> None:
    """Parse the TENSION INDEX section to extract tension records."""
    start, end = _find_section(lines, "TENSION INDEX")
    if start == -1:
        return

    # Pattern: - [Name]: value (SESSION) → value (SESSION STATUS)
    # Or: - [A] → [B]: value (STATUS)
    for i in range(start + 1, end):
        line = lines[i].strip()
        if not line.startswith("-"):
            continue

        # Extract tension name
        name_match = re.search(r'\[([^\]]+)\]', line)
        if not name_match:
            continue

        name = name_match.group(1)

        # Check for arrow pattern: [A] → [B]
        arrow_match = re.search(
            r'\[([^\]]+)\]\s*(?:→|->)\s*\[([^\]]+)\]',
            line,
        )
        if arrow_match:
            name = f"{arrow_match.group(1)}→{arrow_match.group(2)}"

        # Extract all value occurrences
        values = re.findall(r'([\d.]+)\s*\(', line)
        statuses = re.findall(r'\b(RESOLVED|Active|active|NEW)\b', line, re.IGNORECASE)

        initial_value = float(values[0]) if values else 0.0
        current_value = float(values[-1]) if values else 0.0

        # Determine status
        status = "active"
        if statuses:
            last_status = statuses[-1].upper()
            if last_status == "RESOLVED":
                status = "resolved"
            elif last_status == "NEW":
                status = "active"

        # Extract session IDs
        sessions = re.findall(r'(SESSION_\d+)', line)
        created_session = sessions[0] if sessions else None
        resolved_session = sessions[-1] if sessions and status == "resolved" else None

        engine.add_tension(STGTension(
            name=name,
            initial_value=initial_value,
            current_value=current_value,
            status=status,
            created_session=created_session,
            resolved_session=resolved_session,
        ))


def _parse_belief_evolution_index(lines: List[str], engine: STGEngine) -> None:
    """Parse the BELIEF EVOLUTION INDEX section."""
    start, end = _find_section(lines, "BELIEF EVOLUTION INDEX")
    if start == -1:
        return

    # Pattern: - [Old] → [New] (E004, SESSION_008): Description
    pattern = re.compile(
        r'-\s*\[([^\]]+)\]\s*(?:→|->)\s*\[([^\]]+)\]\s*'
        r'\(([^)]*)\):\s*(.*)'
    )

    for i in range(start + 1, end):
        match = pattern.search(lines[i])
        if not match:
            continue

        old_anchor = match.group(1)
        new_anchor = match.group(2)
        provenance = match.group(3)
        description = match.group(4).strip()

        # Extract event_id and session_id from provenance
        event_match = re.search(r'(E\d+)', provenance)
        session_match = re.search(r'(SESSION_\d+)', provenance)

        # Detect level from keywords
        level = 2  # Default
        desc_lower = description.lower()
        if "paradigm" in desc_lower or "level 3" in desc_lower:
            level = 3
        elif "level 1" in desc_lower:
            level = 1

        engine.add_belief_evolution(STGBeliefEvolution(
            old_anchor=old_anchor,
            new_anchor=new_anchor,
            event_id=event_match.group(1) if event_match else None,
            session_id=session_match.group(1) if session_match else None,
            level=level,
            description=description,
        ))


def _parse_events(lines: List[str], engine: STGEngine) -> None:
    """Parse EVENT blocks from the file.

    Handles two formats:
    1. ## EVENT: E0XX - Title
    2. ### E0XX: Title
    """
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Format 1: ## EVENT: E0XX - Title
        event_match1 = re.match(
            r'#{2,3}\s*EVENT:\s*(E\d+)\s*[-–—]\s*(.*)', stripped,
        )
        # Format 2: ### E0XX: Title
        event_match2 = re.match(
            r'#{2,3}\s*(E\d+):\s*(.*)', stripped,
        )

        match = event_match1 or event_match2
        if not match:
            continue

        event_id = match.group(1)
        title = match.group(2).strip()

        # Look ahead for metadata
        importance = 0.5
        event_type = None
        session_id = None
        timestamp = None
        tags: List[str] = []

        # Scan the next 20 lines for metadata fields
        for j in range(i + 1, min(i + 25, len(lines))):
            meta_line = lines[j].strip()

            # Stop at next event or section header
            if re.match(r'#{2,3}\s*(EVENT:|E\d+:|SESSION)', meta_line):
                break
            if meta_line.startswith("---"):
                break

            # Strip markdown bold markers for matching
            clean_line = meta_line.replace("**", "")

            imp_match = re.match(
                r'(?:Importance\s*Score|importance_score)'
                r'\s*[:=]\s*(?:I\s*=\s*)?([\d.]+)',
                clean_line, re.IGNORECASE,
            )
            if imp_match:
                importance = float(imp_match.group(1))

            type_match = re.match(
                r'(?:Event\s*Type|event_type)\s*[:=]\s*["\']?(\w+)',
                clean_line, re.IGNORECASE,
            )
            if type_match:
                event_type = type_match.group(1)

            ts_match = re.match(
                r'(?:Timestamp|timestamp)\s*[:=]\s*["\']?([\d\-T:Z+]+)',
                clean_line, re.IGNORECASE,
            )
            if ts_match:
                timestamp = ts_match.group(1).strip("\"'")

            tag_match = re.match(
                r'(?:Tags|tags)\s*[:=]\s*\[([^\]]*)\]',
                clean_line, re.IGNORECASE,
            )
            if tag_match:
                raw_tags = tag_match.group(1)
                tags = [t.strip().strip("\"'") for t in raw_tags.split(",") if t.strip()]

        # Try to extract session_id from the parent session header
        for j in range(i - 1, max(i - 100, 0), -1):
            sess_match = re.search(r'(SESSION_\d+)', lines[j])
            if sess_match:
                session_id = sess_match.group(1)
                break

        # Extract importance from title if it contains (I=X.XX) or (I = 0.XX)
        title_imp = re.search(r'\(I\s*=\s*([\d.]+)\)', title)
        if title_imp:
            importance = float(title_imp.group(1))
            title = re.sub(r'\s*\(I\s*=\s*[\d.]+\)\s*', ' ', title).strip()

        engine.add_event(STGEvent(
            event_id=event_id,
            session_id=session_id,
            importance_score=importance,
            event_type=event_type,
            title=title,
            timestamp=timestamp,
            tags=tags,
        ))


def _ingest_all_stl(content: str, engine: STGEngine) -> int:
    """Extract and ingest all STL statements from the file.

    Handles both:
    - Code-fenced STL: ```stl ... ``` (via stl_parser's LLM pipeline)
    - Bare STL outside code fences: [A] -> [B] ::mod(...)

    The LLM pipeline automatically handles:
    - Multi-line ::mod() joining
    - Arrow normalization (=>, —>, etc. -> ->)
    - Modifier typo repair (confience -> confidence)
    - Value clamping (confidence=1.5 -> 1.0)

    Returns:
        Number of edges added
    """
    anchor_pattern = re.compile(
        r'\[([^\]]+)\]\s*(?:→|->)\s*\[([^\]]+)\]'
    )

    total = 0

    # Phase 1: Extract STL from code fences using stl_parser's LLM pipeline
    stl_statements = _extract_stl_statements(content)

    current_session = None
    for line in content.split("\n"):
        stripped = line.strip()
        sess_match = re.search(r'(SESSION_\d+)', stripped)
        if sess_match and (
            stripped.startswith("#") or stripped.startswith("scope(")
        ):
            current_session = sess_match.group(1)

    for stmt_str in stl_statements:
        try:
            added = engine.ingest_stl(stmt_str, session_id=current_session)
            total += added
        except Exception:
            try:
                added = engine._ingest_stl_regex(
                    stmt_str, session_id=current_session
                )
                total += added
            except Exception:
                pass

    # Phase 2: Pick up bare STL lines outside code fences
    in_code_fence = False
    current_session = None

    for line in content.split("\n"):
        stripped = line.strip()

        sess_match = re.search(r'(SESSION_\d+)', stripped)
        if sess_match and (
            stripped.startswith("#") or stripped.startswith("scope(")
        ):
            current_session = sess_match.group(1)

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        if in_code_fence:
            continue

        if not anchor_pattern.search(stripped):
            continue

        # Skip index lines (they reference events, not actual STL)
        if stripped.startswith("-") and "::" not in stripped:
            continue

        try:
            added = engine.ingest_stl(stripped, session_id=current_session)
            total += added
        except Exception:
            try:
                added = engine._ingest_stl_regex(
                    stripped, session_id=current_session
                )
                total += added
            except Exception:
                pass

    return total


def _extract_stl_statements(content: str) -> List[str]:
    """Extract complete STL statements from code-fenced blocks.

    Scans for ```stl ... ``` code fences and joins multi-line
    ::mod() blocks into single-line statements.

    Returns:
        List of complete STL statement strings
    """
    statements = []
    lines = content.split("\n")

    in_stl_block = False
    buffer = ""

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```stl"):
            in_stl_block = True
            buffer = ""
            continue
        elif stripped.startswith("```") and in_stl_block:
            if buffer.strip():
                statements.append(buffer.strip())
            in_stl_block = False
            buffer = ""
            continue

        if not in_stl_block:
            continue

        if stripped.startswith("[") and buffer.strip():
            statements.append(buffer.strip())
            buffer = stripped
        elif stripped:
            if buffer:
                buffer += " " + stripped
            else:
                buffer = stripped

    if buffer.strip():
        statements.append(buffer.strip())

    return statements


def get_import_stats(engine: STGEngine) -> Dict:
    """Get summary statistics of an imported memoryMatrix.

    Args:
        engine: STGEngine populated from import

    Returns:
        Dictionary with import statistics
    """
    stats = engine.get_stats()
    return {
        **stats,
        "summary": (
            f"Imported: {stats['node_count']} nodes, "
            f"{stats['edge_count']} edges, "
            f"{stats['session_count']} sessions, "
            f"{stats['event_count']} events, "
            f"{stats['active_tensions']} active tensions, "
            f"{stats['total_tensions']} total tensions"
        ),
    }
