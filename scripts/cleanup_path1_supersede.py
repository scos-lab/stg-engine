"""One-shot cleanup: clear `superseded_at` flags set by the (now removed)
Path 1 duplicate-edge handler.

Background: before this commit, add_edge marked any older edge with the same
(source, target) as superseded whenever the new edge differed in content.
That ignored semantic field — so complementary edges (e.g. action="took"
vs status="had_amazing_time") got wrongly flagged. Path 2
(_flag_suspected_supersede) is the correct, semantically-aware path.

Path-1 footprint: edge has `superseded_at` set but NO `suspected_supersede`.
Path-2 footprint: edge has `suspected_supersede=True` (and usually
`superseded_by`). We only clear Path-1 footprints.

Usage:
    python scripts/cleanup_path1_supersede.py --agent <name> [--apply]
    python scripts/cleanup_path1_supersede.py --path /abs/path/file.stg [--apply]
    python scripts/cleanup_path1_supersede.py --all-agents [--apply]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stg_engine import STGEngine


def cleanup_one(stg_path: Path, apply: bool) -> tuple[int, int]:
    """Returns (path1_cleared, path2_kept)."""
    engine = STGEngine.load(str(stg_path))
    cleared = 0
    kept = 0
    for edge in engine._edges:
        if "superseded_at" not in edge.modifiers:
            continue
        if edge.modifiers.get("suspected_supersede"):
            kept += 1
            continue
        cleared += 1
        if apply:
            del edge.modifiers["superseded_at"]
            edge.modifiers.pop("superseded_by", None)
    if apply and cleared:
        engine.save(str(stg_path))
    return cleared, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", help="agent name under ~/.stg/")
    ap.add_argument("--path", help="absolute .stg file path")
    ap.add_argument("--all-agents", action="store_true",
                    help="iterate every agent under ~/.stg/")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run)")
    args = ap.parse_args()

    home = Path.home() / ".stg"
    targets: list[Path] = []
    if args.path:
        targets.append(Path(args.path))
    if args.agent:
        targets.append(home / args.agent / "memory.stg")
    if args.all_agents:
        for p in sorted(home.iterdir()):
            if p.is_dir():
                f = p / "memory.stg"
                if f.exists():
                    targets.append(f)

    if not targets:
        ap.error("provide --agent / --path / --all-agents")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanning {len(targets)} .stg file(s)...\n")
    total_cleared = 0
    total_kept = 0
    for t in targets:
        if not t.exists():
            print(f"  skip (missing): {t}")
            continue
        cleared, kept = cleanup_one(t, args.apply)
        total_cleared += cleared
        total_kept += kept
        flag = "*" if cleared else " "
        print(f"  {flag} {t.parent.name:<40} path1_to_clear={cleared:<5} "
              f"path2_legitimate={kept}")

    print(f"\nTotal: clear {total_cleared} Path-1 flags, "
          f"keep {total_kept} Path-2 flags.")
    if not args.apply and total_cleared:
        print("Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
