"""Obsidian vault → STL converter.

Reads an Obsidian vault directory and extracts structural relationships
into STL format. No LLM needed — pure programmatic extraction.

Extracts:
  - Directory structure → namespace hierarchy + containment edges
  - [[wikilinks]] → reference edges between notes
  - ## Headings → section nodes with containment edges
  - YAML frontmatter tags → tag classification edges
  - File metadata (mtime) → timestamps

Usage:
    from stg_engine.obsidian_importer import import_obsidian_vault
    stl_text = import_obsidian_vault("/path/to/vault")

    # Or write to file:
    import_obsidian_vault("/path/to/vault", output_path="vault.stl")

CLI:
    python stg_cli.py obsidian /path/to/vault [--output vault.stl] [--ingest]
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from stl_parser import validate_llm_output
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False


# --- Patterns ---

_WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
_TAG_RE = re.compile(r'(?:^|\s)#([A-Za-z\u4e00-\u9fff][A-Za-z0-9_/\u4e00-\u9fff]*)')
_HEADING_RE = re.compile(r'^(#{1,3})\s+(.+)')
_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---', re.DOTALL)
_FRONTMATTER_TAGS_RE = re.compile(r'tags:\s*\[([^\]]*)\]|tags:\s*\n((?:\s*-\s*.+\n)*)')

# Skip patterns — files/dirs that shouldn't be imported
_SKIP_DIRS = {'.obsidian', '.trash', '.git', 'node_modules'}
_SKIP_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.mp3', '.mp4'}


def _sanitize_anchor(text: str) -> str:
    """Convert text to a valid STL anchor name."""
    # Remove markdown formatting
    text = re.sub(r'[*`#\[\](){}!|>]', '', text)
    # Replace spaces/hyphens/dots with underscores
    text = re.sub(r'[\s\-\.]+', '_', text.strip())
    # Remove invalid chars (keep alphanumeric, underscore, colon, CJK)
    text = re.sub(r'[^\w:\u4e00-\u9fff]', '', text)
    # Collapse multiple underscores
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or 'Unnamed'


def _path_to_namespace(rel_path: str) -> str:
    """Convert a relative directory path to a namespace node name.

    Uses underscore-joined path to stay within single-colon namespace limit.
    'algorithm/STL/deque.md' → 'Vault:Algorithm_STL'
    'notes/deque.md' → 'Vault:Notes'
    'deque.md' → 'Vault'
    """
    parts = Path(rel_path).parent.parts
    if not parts or parts == ('.',):
        return 'Vault'
    ns_name = '_'.join(_sanitize_anchor(p) for p in parts)
    return 'Vault:' + ns_name


def _path_to_node_name(rel_path: str) -> str:
    """Convert a relative file path to a node name.

    'algorithm/STL/deque.md' → 'Note:Deque'
    """
    stem = Path(rel_path).stem
    return 'Note:' + _sanitize_anchor(stem)


def _extract_frontmatter(content: str) -> Dict[str, str]:
    """Extract YAML frontmatter as a simple key-value dict."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}
    yaml_block = m.group(1)
    result = {}
    for line in yaml_block.split('\n'):
        line = line.strip()
        if ':' in line and not line.startswith('-'):
            key, _, value = line.partition(':')
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _extract_tags(content: str, frontmatter: Dict[str, str]) -> Set[str]:
    """Extract tags from both frontmatter and inline #tags."""
    tags = set()

    # Frontmatter tags
    fm_tags = frontmatter.get('tags', '')
    if fm_tags:
        # Handle [tag1, tag2] format
        for t in re.split(r'[,\s]+', fm_tags.strip('[]')):
            t = t.strip().strip('"').strip("'").lstrip('#')
            if t and t not in ('include',):  # Skip C++ #include false positives
                tags.add(t)

    # Also try multi-line frontmatter tags
    m = _FRONTMATTER_TAGS_RE.search(content[:500])
    if m:
        block = m.group(1) or m.group(2) or ''
        for t in re.findall(r'[\w\u4e00-\u9fff]+', block):
            if t not in ('include',):
                tags.add(t)

    # Inline #tags (skip code blocks)
    in_code_block = False
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        for m in _TAG_RE.finditer(line):
            tag = m.group(1)
            if tag not in ('include', '1', '2', '3'):  # common false positives
                tags.add(tag)

    return tags


def _extract_wikilinks(content: str) -> Set[str]:
    """Extract [[wikilink]] targets from content."""
    links = set()
    for m in _WIKILINK_RE.finditer(content):
        target = m.group(1).strip()
        # Skip image embeds
        if any(target.lower().endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue
        links.add(target)
    return links


def _extract_headings(content: str) -> List[Tuple[int, str]]:
    """Extract ## and ### headings as (level, title) tuples."""
    headings = []
    in_code_block = False
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if title:
                headings.append((level, title))
    return headings


def _build_stl(source: str, target: str, **mods) -> str:
    """Build a single STL statement.

    Uses manual formatting to support CJK anchor names
    (stl_parser's Anchor validator rejects non-ASCII).
    """
    mod_parts = []
    for k, v in mods.items():
        if isinstance(v, str):
            # Escape inner double quotes
            v_escaped = v.replace('"', '\\"')
            mod_parts.append(f'{k}="{v_escaped}"')
        elif isinstance(v, float):
            mod_parts.append(f'{k}={v}')
        else:
            mod_parts.append(f'{k}={v}')
    mod_str = f' ::mod({", ".join(mod_parts)})' if mod_parts else ''
    return f'[{source}] -> [{target}]{mod_str}'


def scan_vault(vault_path: str) -> List[dict]:
    """Scan an Obsidian vault and return parsed note metadata.

    Returns:
        List of dicts with keys: rel_path, node_name, namespace,
        title, tags, wikilinks, headings, mtime
    """
    vault = Path(vault_path)
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault not found: {vault_path}")

    notes = []

    for md_file in sorted(vault.rglob('*.md')):
        # Skip hidden/special directories
        rel = md_file.relative_to(vault)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue

        try:
            content = md_file.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue

        frontmatter = _extract_frontmatter(content)
        tags = _extract_tags(content, frontmatter)
        wikilinks = _extract_wikilinks(content)
        headings = _extract_headings(content)

        # Title: frontmatter title > first # heading > filename
        title = frontmatter.get('title', '')
        if not title:
            for level, heading in headings:
                if level == 1:
                    title = heading
                    break
        if not title:
            title = md_file.stem

        mtime = datetime.fromtimestamp(md_file.stat().st_mtime)

        notes.append({
            'rel_path': str(rel).replace('\\', '/'),
            'node_name': _path_to_node_name(str(rel)),
            'namespace': _path_to_namespace(str(rel)),
            'title': title,
            'tags': tags,
            'wikilinks': wikilinks,
            'headings': headings,
            'mtime': mtime,
        })

    return notes


def import_obsidian_vault(
    vault_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert an Obsidian vault to STL text.

    Args:
        vault_path: Path to Obsidian vault root directory
        output_path: If provided, write STL to this file

    Returns:
        STL text (all statements joined by newlines)
    """
    notes = scan_vault(vault_path)
    if not notes:
        return '# Empty vault — no .md files found'

    statements: List[str] = []
    statements.append(f'# Obsidian vault: {Path(vault_path).name}')
    statements.append(f'# {len(notes)} notes scanned')
    statements.append(f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    statements.append('')

    # Build wikilink target → node_name lookup
    # Match by filename stem (case-insensitive, Obsidian convention)
    stem_to_node: Dict[str, str] = {}
    for note in notes:
        stem = Path(note['rel_path']).stem.lower()
        stem_to_node[stem] = note['node_name']

    # Collect all unique namespaces for folder hierarchy
    namespaces: Set[str] = set()
    for note in notes:
        namespaces.add(note['namespace'])

    # --- 1. Folder hierarchy edges ---
    statements.append('# === Folder Structure ===')
    emitted_ns = set()
    for ns in sorted(namespaces):
        parts = ns.split(':')
        for i in range(1, len(parts)):
            parent = ':'.join(parts[:i])
            child = ':'.join(parts[:i + 1])
            pair = (parent, child)
            if pair not in emitted_ns:
                emitted_ns.add(pair)
                statements.append(_build_stl(
                    parent, child,
                    rule='definitional', confidence=1.0,
                    description=f'folder contains subfolder',
                ))

    # --- 2. Note nodes + containment in namespace ---
    statements.append('')
    statements.append('# === Notes ===')
    for note in notes:
        # Note → Namespace (containment)
        statements.append(_build_stl(
            note['namespace'], note['node_name'],
            rule='definitional', confidence=1.0,
            description=note['title'],
            path=note['rel_path'],
            timestamp=note['mtime'].strftime('%Y-%m-%d'),
        ))

    # --- 3. Wikilink edges ---
    link_count = 0
    statements.append('')
    statements.append('# === Wikilinks ===')
    for note in notes:
        for link_target in note['wikilinks']:
            target_stem = link_target.lower().strip()
            target_node = stem_to_node.get(target_stem)
            if target_node and target_node != note['node_name']:
                statements.append(_build_stl(
                    note['node_name'], target_node,
                    rule='logical', confidence=0.85,
                    description=f'references via [[{link_target}]]',
                ))
                link_count += 1

    # --- 4. Tag edges ---
    all_tags: Set[str] = set()
    for note in notes:
        all_tags.update(note['tags'])

    if all_tags:
        statements.append('')
        statements.append('# === Tags ===')
        for note in notes:
            for tag in sorted(note['tags']):
                tag_node = 'Tag:' + _sanitize_anchor(tag)
                statements.append(_build_stl(
                    note['node_name'], tag_node,
                    rule='logical', confidence=0.80,
                    description=f'tagged #{tag}',
                ))

    # --- 5. Section headings (top-level ## only, to avoid noise) ---
    section_count = 0
    statements.append('')
    statements.append('# === Sections ===')
    for note in notes:
        h2_headings = [(lvl, title) for lvl, title in note['headings'] if lvl == 2]
        if len(h2_headings) > 15:
            # Too many sections — skip to avoid noise
            continue
        for _, title in h2_headings:
            section_name = note['node_name'] + '_' + _sanitize_anchor(title)
            statements.append(_build_stl(
                note['node_name'], section_name,
                rule='definitional', confidence=0.95,
                description=title,
            ))
            section_count += 1

    # --- Summary ---
    statements.insert(3, f'# {link_count} wikilinks, {len(all_tags)} tags, {section_count} sections')

    raw_stl = '\n'.join(statements) + '\n'

    # Run through validate_llm_output for auto-repair
    # (fixes multi-colon anchors, unquoted values, etc.)
    if HAS_VALIDATOR:
        result = validate_llm_output(raw_stl)
        stl_text = result.cleaned_text
        if result.repairs:
            repair_summary = f'# Auto-repairs applied: {len(result.repairs)}'
            stl_text = repair_summary + '\n' + stl_text
    else:
        stl_text = raw_stl

    if output_path:
        Path(output_path).write_text(stl_text, encoding='utf-8')

    return stl_text
