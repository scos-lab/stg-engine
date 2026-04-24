# STG Engine Optimization & Bug Report

**Originally reported:** 2026-04-24
**Reporter context:** Bible-NWT STG Pilot Research (Gemini + wuko + Syn-claude)
**Scope:** `stg-engine` on `bible-nwt` library (~5.3k nodes, ~61k edges)

> Internal in-repo tracker. Commits referencing an issue number here should cite `OPTIMIZATION_LOG #N`.
> GitHub issues are opened only for items with user-facing behavior changes (marked below).

---

### 1. Hardcoded Description Truncation (Medium)

- **Issue:** In `stg_engine/cli.py`, the `description` modifier is truncated to 150 characters (`[:150]`) in node/query displays.
- **Impact:** Detailed study-note descriptions (common in research use) are partially unreadable via CLI; forces direct API calls.
- **Location:** `stg_engine/cli.py` (inside `cmd_node`, `cmd_query`, and some helpers).
- **Fix:** Add `--full` flag; default behavior preserved for UX, `--full` bypasses truncation.
- **Status:** 🟢 **Fixed** 2026-04-24 (in-repo commit; no GitHub issue needed — internal UX change).

### 2. Module Path Resolution (Low / DX)

- **Issue:** `python3 -m stg_engine.cli` fails without `PYTHONPATH` set. Normal users should never hit this because `stg` alias + editable pip install handle it, but direct module invocation is inconsistent.
- **Impact:** Increases friction for automated agents / ephemeral environments.
- **Fix direction:** Verify `pyproject.toml` entry point; ensure editable install is fully self-contained.
- **Status:** 🟡 **Deferred** — GitHub issue opened for packaging review (see `Related issues` below).

### 3. Missing CLI Parameter Documentation (Low)

- **Issue:** `--path <db.stg>` is functional but not in `--help` output.
- **Impact:** Agents rely on environment variables; multi-database workflows harder to script.
- **Fix:** Add `--path` to the main CLI usage help.
- **Status:** 🟢 **Fixed** 2026-04-24 (in-repo commit).

### 4. Data Integrity / Index Misalignment — **NOT an stg-engine bug**

- **Issue:** Querying `Verse_John_3_3` returned text associated with John 3:1 (Nicodemus introduction).
- **Root cause (found on investigation):** This was a **downstream pilot extractor bug**, not stg-engine's fault. The JWPUB source SQLite uses two CSS classes for verse labels:
  - `class="vl"` = verse label ("1", "2", "3", …)
  - `class="cl"` = chapter label (the big chapter number at verse 1 position)
  The Bible pilot's `extract_json.py` used `strip_html(label)` without distinguishing, so every verse 1 of each chapter inherited the chapter number as its verse number, overwriting `Verse_John_N_N` with chapter-1's text.
- **Impact:** 42 verses (21 chapters × 2 verses each) affected in `bible-nwt`; limited to verse text — no impact on cross-refs, outlines, study notes, or semantic edges.
- **Status:** 🟡 **Being fixed** in the pilot (not here). See `research/bible-stg-pilot/` for patch script.

### 5. Propagate Keyword Sensitivity (Medium)

- **Issue:** Natural-language queries like `"overview"` return `No nodes activated`.
- **Impact:** Long-term memory feature feels brittle for vague queries.
- **Fix direction:** Seed-selection fallback — if exact token matching yields zero seeds, try case-insensitive substring match against node names; emit a "did you mean?" hint.
- **Status:** 🟡 **Open** — GitHub issue opened for design discussion before implementation (behavioral change).

### 6. Output Volume / Virtual Edge Noise (Medium)

- **Issue:** `stg node Jesus` produces 2963 lines because virtual edges (~150 per source) dominate output.
- **Impact:** Hard to read in terminal; agents must filter.
- **Fix:** Default `cmd_node` to show only `knowledge` edges; add `--virtual` flag to include computed sibling/co-source associations.
- **Status:** 🟢 **Fixed** 2026-04-24 (in-repo commit; behavior change, documented in CLI help).

---

## Summary Table

| # | Severity | Resolution |
|---|----------|-----------|
| 1 | Medium | 🟢 Fixed via `--full` flag |
| 2 | Low | 🟡 GitHub issue for packaging review |
| 3 | Low | 🟢 Fixed (help text) |
| 4 | — (not engine bug) | 🟡 Fixed in pilot downstream |
| 5 | Medium | 🟡 GitHub issue for design review |
| 6 | Medium | 🟢 Fixed via `--virtual` opt-in |

## Related GitHub Issues

- #1 — Packaging: `python3 -m stg_engine.cli` requires PYTHONPATH (Issue #2 from this log)
- #2 — Propagate: natural-language queries return zero activation with no fallback (Issue #5 from this log)

---

## Methodology Note

This file is an **in-repo change log**, not a replacement for GitHub issues. We use it for:

- Internal rationale tracking (why a 1-line change was made)
- Batch audit when a researcher discovers multiple items at once
- Traceability for commits that don't warrant their own issue

GitHub issues are reserved for items where external users/developers benefit from public discussion — typically behavior changes or items needing design input before implementation.
