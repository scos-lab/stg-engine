"""STG World Modeling — discover rules by observing change.

Phase 13: World modeling through Δρ observation.

Core loop: observe change → infer rules → predict → verify → explore efficiently.

变化即信息。规则即 V(ρ)。世界模型即对 ρ 场演化规律的理解。

Usage:
    report = analyze_change(grid_before, grid_after, action_id, level_before, level_after)
    new_rules = infer_rules_from_observations(engine, observations)
    prediction = predict_action_effect(engine, current_state, action_id)
    error = verify_and_update(engine, prediction, actual)
    action = select_action_by_information_gain(engine, state, available)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ═══════════════════════════════════════════════════════════════════
# ChangeReport — structured Δρ observation
# ═══════════════════════════════════════════════════════════════════

ACTION_NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACTION", 6: "CLICK", 7: "UNDO"}


@dataclass
class ChangeReport:
    """Structured observation of what changed after an action (Δρ)."""

    action_id: int
    action_name: str

    # Quantitative change
    pixels_changed: int
    level_before: int
    level_after: int
    level_up: bool

    # Movement detection
    movement_vector: Optional[Tuple[int, int]] = None  # (dx, dy)
    moved_object_color: Optional[int] = None
    moved_object_size: int = 0

    # State classification
    is_blocked: bool = False       # pixels_changed < threshold (action had no real effect)
    is_new_state: bool = False     # frame hash changed
    is_meaningful: bool = False    # significant pixel change (> noise threshold)

    # Cost tracking
    yellow_bar_delta: int = 0      # change in progress bar

    # Frame identity
    frame_before_hash: str = ""
    frame_after_hash: str = ""

    # Raw changed pixels (capped for memory)
    changed_pixels: List[Dict[str, Any]] = field(default_factory=list)


def analyze_change(
    grid_before: List[List[int]],
    grid_after: List[List[int]],
    action_id: int,
    level_before: int = 0,
    level_after: int = 0,
    noise_threshold: int = 10,
) -> ChangeReport:
    """Compare two frames and produce a structured change report.

    This is the Δρ detector — the fundamental observation operation.

    Args:
        grid_before: frame before action
        grid_after: frame after action
        action_id: action that was taken
        level_before: game level before action
        level_after: game level after action
        noise_threshold: pixel changes below this are considered noise/blocked

    Returns:
        ChangeReport with structured change analysis
    """
    from stg_engine.perception import grid_hash

    h = len(grid_before)
    w = len(grid_before[0]) if grid_before else 0

    # Pixel-level diff
    changed = []
    for y in range(h):
        for x in range(w):
            if grid_before[y][x] != grid_after[y][x]:
                changed.append({
                    "x": x, "y": y,
                    "old": grid_before[y][x],
                    "new": grid_after[y][x],
                })

    pixels_changed = len(changed)
    level_up = level_after > level_before
    is_blocked = pixels_changed < noise_threshold and not level_up
    hash_before = grid_hash(grid_before)
    hash_after = grid_hash(grid_after)
    is_new_state = hash_before != hash_after
    is_meaningful = pixels_changed >= noise_threshold

    # Movement detection: find the dominant object that moved
    movement_vector = None
    moved_color = None
    moved_size = 0

    if is_meaningful and changed:
        # Group changes by "disappeared" (old color != background) and "appeared" (new color != background)
        # Detect background as most common color in grid
        all_colors = Counter(c for row in grid_before for c in row)
        bg_color = all_colors.most_common(1)[0][0]

        disappeared = [(c["x"], c["y"]) for c in changed if c["old"] != bg_color and c["new"] == bg_color]
        appeared = [(c["x"], c["y"]) for c in changed if c["new"] != bg_color and c["old"] == bg_color]

        if disappeared and appeared:
            # Center of mass shift = movement vector
            old_cx = sum(p[0] for p in disappeared) / len(disappeared)
            old_cy = sum(p[1] for p in disappeared) / len(disappeared)
            new_cx = sum(p[0] for p in appeared) / len(appeared)
            new_cy = sum(p[1] for p in appeared) / len(appeared)

            dx = int(round(new_cx - old_cx))
            dy = int(round(new_cy - old_cy))

            if abs(dx) > 0 or abs(dy) > 0:
                movement_vector = (dx, dy)

            # Identify moved object color (most common new color in appeared region)
            new_colors = Counter(grid_after[y][x] for x, y in appeared if grid_after[y][x] != bg_color)
            if new_colors:
                moved_color = new_colors.most_common(1)[0][0]
                moved_size = len(appeared)

    # Yellow bar tracking: count color 11 (yellow) pixels in bottom rows
    def count_yellow(grid):
        return sum(1 for y in range(max(0, len(grid) - 4), len(grid))
                   for x in range(len(grid[0])) if grid[y][x] == 11)

    yellow_before = count_yellow(grid_before)
    yellow_after = count_yellow(grid_after)
    yellow_delta = yellow_after - yellow_before

    return ChangeReport(
        action_id=action_id,
        action_name=ACTION_NAMES.get(action_id, f"ACTION{action_id}"),
        pixels_changed=pixels_changed,
        level_before=level_before,
        level_after=level_after,
        level_up=level_up,
        movement_vector=movement_vector,
        moved_object_color=moved_color,
        moved_object_size=moved_size,
        is_blocked=is_blocked,
        is_new_state=is_new_state,
        is_meaningful=is_meaningful,
        yellow_bar_delta=yellow_delta,
        frame_before_hash=hash_before,
        frame_after_hash=hash_after,
        changed_pixels=changed[:50],  # cap for memory
    )


# ═══════════════════════════════════════════════════════════════════
# Rule Inference — automatic pattern detection from observations
# ═══════════════════════════════════════════════════════════════════


def infer_rules_from_observations(
    observations: List[ChangeReport],
    existing_rules: Optional[Dict[str, Dict]] = None,
) -> List[str]:
    """Automatically infer causal rules from accumulated observations.

    No LLM needed. Pure pattern detection:
    - Same action → consistent movement → movement rule
    - Action → 0 change → blocked rule
    - Action → level_up → win condition evidence
    - Yellow bar always decreases → cost rule

    Args:
        observations: list of ChangeReports from gameplay
        existing_rules: already known rules (to avoid duplicates)

    Returns:
        List of STL causal rule statements
    """
    if not observations:
        return []

    rules = []
    existing = existing_rules or {}

    # 1. Movement rules: group by action, find consistent movement vectors
    action_movements: Dict[int, List[Optional[Tuple[int, int]]]] = defaultdict(list)
    action_blocked: Dict[int, int] = defaultdict(int)
    action_total: Dict[int, int] = defaultdict(int)

    for obs in observations:
        action_total[obs.action_id] += 1
        if obs.is_blocked:
            action_blocked[obs.action_id] += 1
        if obs.movement_vector:
            action_movements[obs.action_id].append(obs.movement_vector)

    for action_id, vectors in action_movements.items():
        if not vectors:
            continue
        action_name = ACTION_NAMES.get(action_id, f"ACTION{action_id}")

        # Find most common movement vector for this action
        vec_counts = Counter(vectors)
        most_common_vec, count = vec_counts.most_common(1)[0]
        dx, dy = most_common_vec
        total = action_total[action_id]
        consistency = count / total if total > 0 else 0
        conf = min(0.95, 0.5 + consistency * 0.45)

        rule_key = f"movement_{action_name}"
        if rule_key not in existing and consistency > 0.3:
            rules.append(
                f'[Action:{action_name}] -> [Effect:Move_dx{dx:+d}_dy{dy:+d}] '
                f'::mod(rule="causal", confidence={conf:.2f}, '
                f'strength={consistency:.2f}, '
                f'dx="{dx}", dy="{dy}", '
                f'verified_count={count}, total_observations={total}, '
                f'source="auto_inference")'
            )

    # 2. Blocked rules: actions that are often blocked
    for action_id, blocked_count in action_blocked.items():
        total = action_total[action_id]
        if blocked_count == 0:
            continue
        action_name = ACTION_NAMES.get(action_id, f"ACTION{action_id}")
        block_rate = blocked_count / total

        rule_key = f"blocked_{action_name}"
        if rule_key not in existing and block_rate > 0.2:
            conf = min(0.90, 0.5 + block_rate * 0.4)
            rules.append(
                f'[Action:{action_name}] -> [Effect:Sometimes_Blocked] '
                f'::mod(rule="causal", confidence={conf:.2f}, '
                f'block_rate="{block_rate:.2f}", '
                f'blocked_count={blocked_count}, total={total}, '
                f'source="auto_inference")'
            )

    # 3. Cost rule: yellow bar change
    yellow_deltas = [obs.yellow_bar_delta for obs in observations if obs.yellow_bar_delta != 0]
    if yellow_deltas:
        common_delta = Counter(yellow_deltas).most_common(1)[0]
        delta_val, delta_count = common_delta
        if delta_count >= 3 and "cost_yellow" not in existing:
            rules.append(
                f'[Action:Any] -> [Cost:Yellow_Bar_Delta_{delta_val:+d}] '
                f'::mod(rule="causal", confidence=0.95, '
                f'delta="{delta_val}", '
                f'implication="finite action budget", '
                f'verified_count={delta_count}, '
                f'source="auto_inference")'
            )

    # 4. Level up evidence
    level_ups = [obs for obs in observations if obs.level_up]
    if level_ups:
        for obs in level_ups:
            rules.append(
                f'[State:Frame_{obs.frame_before_hash}] -> [Effect:Level_Up] '
                f'::mod(rule="causal", confidence=0.98, '
                f'action="{obs.action_name}", '
                f'pixels_changed={obs.pixels_changed}, '
                f'source="direct_observation")'
            )

    return rules


# ═══════════════════════════════════════════════════════════════════
# Prediction & Verification
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Prediction:
    """Predicted outcome of an action based on known rules."""
    action_id: int
    expected_blocked: bool = False
    expected_movement: Optional[Tuple[int, int]] = None
    expected_pixels_changed: int = 0
    expected_level_up: bool = False
    confidence: float = 0.0
    based_on_rules: List[str] = field(default_factory=list)


def predict_action_effect(
    engine: "STGEngine",
    current_hash: str,
    action_id: int,
    observations: List[ChangeReport],
) -> Prediction:
    """Predict what will happen if action is taken, based on known rules.

    Uses historical observations for this action to predict outcome.
    """
    action_name = ACTION_NAMES.get(action_id, f"ACTION{action_id}")

    # Gather historical data for this action
    relevant = [o for o in observations if o.action_id == action_id]

    if not relevant:
        return Prediction(action_id=action_id, confidence=0.0)

    # Most common movement vector
    vectors = [o.movement_vector for o in relevant if o.movement_vector]
    blocked_count = sum(1 for o in relevant if o.is_blocked)
    total = len(relevant)

    pred = Prediction(action_id=action_id)

    if blocked_count > total * 0.7:
        pred.expected_blocked = True
        pred.confidence = blocked_count / total
        pred.based_on_rules.append(f"blocked_{action_name}")
    elif vectors:
        vec_counts = Counter(vectors)
        best_vec, count = vec_counts.most_common(1)[0]
        pred.expected_movement = best_vec
        pred.expected_pixels_changed = int(np.mean([o.pixels_changed for o in relevant if o.is_meaningful]))
        pred.confidence = count / total
        pred.based_on_rules.append(f"movement_{action_name}")

    return pred


def verify_and_update(
    prediction: Prediction,
    actual: ChangeReport,
    engine: Optional["STGEngine"] = None,
) -> float:
    """Compare prediction with actual observation. Returns prediction error.

    Error: 0.0 = perfect prediction, 1.0 = completely wrong.

    Also updates rule confidence in STG if engine is provided.
    """
    error = 0.0
    checks = 0

    # Check blocked prediction
    if prediction.confidence > 0:
        if prediction.expected_blocked:
            checks += 1
            if not actual.is_blocked:
                error += 1.0  # predicted blocked but moved
        elif prediction.expected_movement:
            checks += 1
            if actual.is_blocked:
                error += 1.0  # predicted movement but blocked
            elif actual.movement_vector:
                # Compare direction
                pred_dx, pred_dy = prediction.expected_movement
                act_dx, act_dy = actual.movement_vector
                if pred_dx != act_dx or pred_dy != act_dy:
                    error += 0.5  # moved but wrong direction/magnitude

    if checks == 0:
        return 0.5  # no prediction to verify

    normalized_error = error / checks

    # Update rule confidence in STG
    if engine is not None and prediction.based_on_rules:
        for rule_name in prediction.based_on_rules:
            # Find matching edges and adjust confidence
            for edge in engine._edges:
                if rule_name in str(edge.source) or rule_name in str(edge.target):
                    if normalized_error < 0.2:
                        edge.confidence = min(0.99, edge.confidence + 0.02)
                    elif normalized_error > 0.5:
                        edge.confidence = max(0.1, edge.confidence - 0.05)

    return normalized_error


# ═══════════════════════════════════════════════════════════════════
# Information-Gain Exploration
# ═══════════════════════════════════════════════════════════════════


def select_action_by_information_gain(
    current_hash: str,
    available_actions: List[int],
    observations: List[ChangeReport],
    state_action_history: Optional[Dict[str, set]] = None,
) -> Tuple[int, str]:
    """Select action that maximizes expected information gain.

    Priority:
    1. Untried (state, action) pairs — maximum uncertainty
    2. Actions with inconsistent results — need more data
    3. Actions that led to new states — expand exploration frontier
    4. Avoid: well-understood (state, action) pairs, known-blocked actions

    Args:
        current_hash: hash of current frame
        available_actions: valid action IDs
        observations: historical observations
        state_action_history: {frame_hash: set of tried action_ids}

    Returns:
        (action_id, reason)
    """
    import random

    if not available_actions:
        return 0, "no_actions"

    history = state_action_history or {}
    tried_here = history.get(current_hash, set())

    # 1. Untried actions at this state — maximum information gain
    untried = [a for a in available_actions if a not in tried_here]
    if untried:
        action = untried[0]  # systematic, not random
        return action, f"untried_at_state(tried={len(tried_here)}/{len(available_actions)})"

    # 2. Actions with low observation count (globally) — need more data
    action_counts = Counter(o.action_id for o in observations)
    least_tried = min(available_actions, key=lambda a: action_counts.get(a, 0))
    if action_counts.get(least_tried, 0) < 3:
        return least_tried, f"low_data(n={action_counts.get(least_tried, 0)})"

    # 3. Actions with inconsistent results — prediction uncertainty
    action_consistency: Dict[int, float] = {}
    for a in available_actions:
        relevant = [o for o in observations if o.action_id == a]
        if len(relevant) < 2:
            action_consistency[a] = 0.0  # unknown = interesting
            continue
        vectors = [o.movement_vector for o in relevant if o.movement_vector]
        if vectors:
            vec_counts = Counter(vectors)
            _, top_count = vec_counts.most_common(1)[0]
            action_consistency[a] = top_count / len(relevant)
        else:
            action_consistency[a] = 0.5

    least_consistent = min(available_actions, key=lambda a: action_consistency.get(a, 0))
    if action_consistency.get(least_consistent, 1.0) < 0.7:
        return least_consistent, f"inconsistent(consistency={action_consistency[least_consistent]:.2f})"

    # 4. Actions that have led to new states in the past — frontier expansion
    frontier_actions = []
    for a in available_actions:
        relevant = [o for o in observations if o.action_id == a and o.is_meaningful]
        if relevant:
            frontier_actions.append((a, len(relevant)))
    if frontier_actions:
        frontier_actions.sort(key=lambda x: x[1], reverse=True)
        return frontier_actions[0][0], "frontier_expansion"

    # 5. Fallback: random
    return random.choice(available_actions), "random_fallback"


# ═══════════════════════════════════════════════════════════════════
# World Modeling Step — the complete loop
# ═══════════════════════════════════════════════════════════════════


def world_modeling_step(
    engine: "STGEngine",
    grid: List[List[int]],
    available_actions: List[int],
    prev_grid: Optional[List[List[int]]] = None,
    prev_action: Optional[int] = None,
    prev_level: int = 0,
    current_level: int = 0,
    observations: Optional[List[ChangeReport]] = None,
    state_action_history: Optional[Dict[str, set]] = None,
    adapter: Optional[Any] = None,
    step_number: int = 0,
    game_id: Optional[str] = None,
) -> Tuple[int, str, Optional[ChangeReport]]:
    """One step of the world modeling loop.

    OBSERVE → INFER → PREDICT → SELECT → (caller executes, then calls again)

    Args:
        engine: STGEngine
        grid: current game frame
        available_actions: valid actions
        prev_grid: previous frame (None on first step)
        prev_action: action taken last step
        prev_level: level before last action
        current_level: level now
        observations: accumulated ChangeReports (mutable, appended to)
        state_action_history: {hash: set(action_ids)} (mutable, updated)
        adapter: optional LLM adapter
        step_number: current step
        game_id: game identifier

    Returns:
        (action_id, reason, change_report_from_prev_action)
    """
    from stg_engine.perception import grid_hash, perceive_frame

    if observations is None:
        observations = []
    if state_action_history is None:
        state_action_history = {}

    # Use gameplay-area hash (exclude bottom HUD rows where yellow bar changes)
    # This prevents yellow bar changes from making every frame "unique"
    gameplay_grid = grid[:-4] if len(grid) > 4 else grid
    current_hash = grid_hash(gameplay_grid)
    change_report = None

    # 1. OBSERVE: analyze what changed from previous action
    if prev_grid is not None and prev_action is not None:
        change_report = analyze_change(
            prev_grid, grid, prev_action,
            level_before=prev_level, level_after=current_level,
        )
        observations.append(change_report)

        # Record in state_action_history (exclude HUD)
        prev_gameplay = prev_grid[:-4] if len(prev_grid) > 4 else prev_grid
        prev_hash = grid_hash(prev_gameplay)
        if prev_hash not in state_action_history:
            state_action_history[prev_hash] = set()
        state_action_history[prev_hash].add(prev_action)

        # If action was blocked, we're still at the same state
        # Use prev_hash as current_hash (the tiny pixel change doesn't matter)
        if change_report.is_blocked:
            current_hash = prev_hash

    # 2. INFER: auto-detect rules from accumulated observations
    if len(observations) >= 3 and len(observations) % 5 == 0:
        # Every 5 steps, re-infer rules
        new_rules = infer_rules_from_observations(observations)
        for rule_stl in new_rules:
            try:
                engine.ingest_stl(rule_stl)
            except Exception:
                pass

    # 3. PREDICT & VERIFY
    if prev_action is not None and change_report is not None and len(observations) >= 3:
        pred = predict_action_effect(engine, current_hash, prev_action, observations)
        if pred.confidence > 0.3:
            verify_and_update(pred, change_report, engine)

    # 4. Anti-oscillation: detect if recent actions are dominated by 2 directions
    import random
    force_new = False
    if len(observations) >= 6:
        recent_actions = [o.action_id for o in observations[-6:]]
        action_freq = Counter(recent_actions)
        dominant = action_freq.most_common(2)
        # If top 2 actions account for 5+ of last 6 moves → stuck in corridor
        if len(dominant) >= 2 and dominant[0][1] + dominant[1][1] >= 5:
            dominant_ids = {dominant[0][0], dominant[1][0]}
            other_actions = [a for a in available_actions if a not in dominant_ids]
            if other_actions:
                action_id = other_actions[0]
                reason = f"anti_oscillation(break_{ACTION_NAMES.get(dominant[0][0],'?')}/{ACTION_NAMES.get(dominant[1][0],'?')})"
                force_new = True

    if not force_new:
        # 5. SELECT: choose action by information gain
        action_id, reason = select_action_by_information_gain(
            current_hash, available_actions, observations, state_action_history,
        )

    # Perceive frame in STG (maintains perception index)
    perceive_frame(engine, grid, game_id=game_id, step_number=step_number, level=current_level)

    return action_id, reason, change_report


# ═══════════════════════════════════════════════════════════════════
# Self-Aware Prompting — STG writes its own questions
# ═══════════════════════════════════════════════════════════════════


def build_self_aware_prompt(
    observations: List[ChangeReport],
    known_rules: List[str],
    grid_before: Optional[List[List[int]]] = None,
    grid_after: Optional[List[List[int]]] = None,
    last_action: Optional[int] = None,
    trigger_reason: str = "routine",
) -> str:
    """STG generates its own question based on its knowledge state.

    Not a fixed template. The prompt reflects what STG knows,
    what surprised it, and what it needs to learn.

    Args:
        observations: accumulated ChangeReports
        known_rules: list of already-inferred STL rule strings
        grid_before: frame before last action (for showing Δρ)
        grid_after: frame after last action
        last_action: action that was just taken
        trigger_reason: why LLM is being called (anomaly/stuck/initial)

    Returns:
        Natural language prompt for LLM
    """
    from stg_engine.perception import _grid_to_text

    sections = []

    # === What I know ===
    if known_rules:
        sections.append("## What I know (verified rules):")
        for rule in known_rules[:10]:
            # Extract key info from STL
            sections.append(f"- {rule[:150]}")
    else:
        sections.append("## What I know: Nothing yet. First observation.")

    # === What I've observed ===
    if observations:
        n_total = len(observations)
        n_blocked = sum(1 for o in observations if o.is_blocked)
        n_meaningful = sum(1 for o in observations if o.is_meaningful)
        n_level_up = sum(1 for o in observations if o.level_up)

        sections.append(f"\n## What I've done ({n_total} actions):")
        sections.append(f"- Meaningful moves: {n_meaningful}")
        sections.append(f"- Blocked (hit wall): {n_blocked}")
        sections.append(f"- Level ups: {n_level_up}")

        # Action summary
        action_summary = Counter(o.action_name for o in observations)
        sections.append(f"- Actions tried: {dict(action_summary)}")

        # Movement vectors observed
        vectors = [(o.action_name, o.movement_vector) for o in observations if o.movement_vector]
        if vectors:
            vec_summary = Counter(vectors)
            sections.append("- Movement patterns:")
            for (act, vec), count in vec_summary.most_common(5):
                sections.append(f"  {act}: dx={vec[0]}, dy={vec[1]} ({count} times)")

    # === What surprised me (trigger context) ===
    if trigger_reason == "anomaly" and observations:
        last_obs = observations[-1]
        sections.append(f"\n## What surprised me:")
        sections.append(f"- Action {last_obs.action_name} produced {last_obs.pixels_changed}px change")
        sections.append(f"- This is unexpected based on my rules")
        if last_obs.movement_vector:
            sections.append(f"- Movement: {last_obs.movement_vector}")

    elif trigger_reason == "stuck":
        sections.append(f"\n## Why I'm stuck:")
        sections.append(f"- I've been oscillating between the same positions")
        sections.append(f"- All directions I try either repeat or get blocked")

    elif trigger_reason == "initial":
        sections.append(f"\n## First look at this game:")
        sections.append(f"- I need to understand the basic rules and goal")

    # === The actual change to analyze ===
    if grid_before is not None and grid_after is not None and last_action is not None:
        action_name = ACTION_NAMES.get(last_action, f"ACTION{last_action}")
        sections.append(f"\n## Latest observation (action: {action_name}):")
        sections.append("BEFORE:")
        sections.append(_grid_to_text(grid_before))
        sections.append("AFTER:")
        sections.append(_grid_to_text(grid_after))

    # === What I need to know ===
    sections.append("\n## What I need from you:")

    if not known_rules:
        sections.append("- What are the basic movement rules?")
        sections.append("- What objects are on the grid?")
        sections.append("- What might be the goal?")
    elif trigger_reason == "stuck":
        sections.append("- I understand basic movement but can't progress.")
        sections.append("- What am I missing? Is there a hidden mechanic?")
        sections.append("- How do I reach the area I can't get to?")
    elif trigger_reason == "anomaly":
        sections.append("- Something unexpected happened. What rule explains this?")
    else:
        sections.append("- What new rules can you infer from this observation?")
        sections.append("- Any patterns I'm missing?")

    sections.append("\nRespond ONLY with STL causal rules. No thinking. Just output.")
    sections.append("Example: [Action:UP] -> [Effect:Block_Moves_5px] ::mod(rule=\"causal\", confidence=0.9)")

    return "\n".join(sections)


async def ask_llm_self_aware(
    engine: "STGEngine",
    adapter: Any,
    observations: List[ChangeReport],
    grid_before: Optional[List[List[int]]] = None,
    grid_after: Optional[List[List[int]]] = None,
    last_action: Optional[int] = None,
    trigger_reason: str = "routine",
) -> List[str]:
    """STG asks LLM a self-generated question and ingests the answer.

    The question is generated from STG's own knowledge state —
    what it knows, what confused it, what it needs.

    Returns: list of STL statements ingested
    """
    try:
        from skc.types.llm import Message
    except ImportError:
        raise NotImplementedError("This LLM-dependent feature requires the SKC package. Use stg-engine for the core algorithms; SKC for LLM integration.")
    from stl_parser import validate_llm_output

    # Gather known rules from STG
    known_rules = []
    for edge in engine._edges:
        if (edge.source.startswith("Action:") or edge.source.startswith("State:")) \
                and edge.rule == "causal":
            mods = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in edge.modifiers.items()
                if k not in ("_created_at",) and v is not None
            )
            known_rules.append(f"[{edge.source}] -> [{edge.target}] ::mod({mods})")

    # STG generates its own prompt
    prompt = build_self_aware_prompt(
        observations, known_rules,
        grid_before, grid_after, last_action,
        trigger_reason,
    )

    # Call LLM
    messages = [Message(role="user", content=prompt)]
    response = await adapter.generate_response(
        messages,
        system_prompt=(
            "You are analyzing a game environment. "
            "The player is an AI that can only move UP/DOWN/LEFT/RIGHT. "
            "Based on what the AI knows and doesn't know, infer game rules. "
            "Output ONLY STL causal rules, one per line. No thinking. Just output.\n\n"
            "STRICT STL syntax rules:\n"
            "- Each line: [Source] -> [Target] ::mod(key=\"value\")\n"
            "- Anchors: [PascalCase_Name] — letters, digits, underscores ONLY. NO commas, equals, dots, spaces, AND, OR\n"
            "- NEVER write [A] AND [B] -> [C]. Instead: [A_When_B] -> [C]\n"
            "- Use description= in ::mod() for complex conditions and details\n"
            "Examples:\n"
            "[Action_UP] -> [Effect_Move_North] ::mod(rule=\"causal\", confidence=0.9, description=\"Player moves up 4px when path is clear\")\n"
            "[Tile_1] -> [Property_Impassable] ::mod(rule=\"definitional\", confidence=0.95, description=\"Color 1 blocks movement\")\n"
            "[Action_UP_Blocked_By_Tile3] -> [Effect_No_Movement] ::mod(rule=\"causal\", confidence=0.9, description=\"UP blocked when tile 3 is above\")\n"
        ),
    )

    # Parse and ingest
    # Pre-clean: fix common LLM syntax errors before parsing
    import re
    raw_text = response.content
    # Fix [A] AND [B] -> [C] pattern → [A_AND_B] -> [C]
    raw_text = re.sub(
        r'\[([^\]]+)\]\s+AND\s+\[([^\]]+)\]',
        lambda m: f'[{m.group(1)}_AND_{m.group(2)}]',
        raw_text,
    )
    # Fix [A] OR [B] -> [C] pattern
    raw_text = re.sub(
        r'\[([^\]]+)\]\s+OR\s+\[([^\]]+)\]',
        lambda m: f'[{m.group(1)}_OR_{m.group(2)}]',
        raw_text,
    )

    logger.debug(f"[LLM RAW] {response.content[:500]}")
    result = validate_llm_output(raw_text)
    stl_statements = []
    if result.statements:
        for line in result.cleaned_text.strip().split("\n"):
            line = line.strip()
            if not line or "->" not in line:
                continue
            stl_statements.append(line)
            try:
                engine.ingest_stl(line)
            except Exception as e:
                logger.debug(f"[INGEST FAIL] {line[:100]}: {e}")
    else:
        logger.debug(f"[LLM PARSE FAIL] errors={result.errors}, raw_len={len(response.content)}")

    return stl_statements


def _render_grid_to_base64(grid: List[List[int]]) -> str:
    """Render grid to PNG and return as base64 string for vision LLM."""
    import base64
    import io
    try:
        from PIL import Image
    except ImportError:
        return ""

    # ARC color palette
    COLORS = {
        0: (0, 0, 0), 1: (255, 255, 255), 2: (255, 0, 0),
        3: (0, 255, 0), 4: (68, 68, 68), 5: (0, 200, 200),
        6: (255, 0, 255), 7: (255, 255, 0), 8: (255, 128, 0),
        9: (30, 130, 255), 10: (200, 230, 255), 11: (255, 210, 0),
        12: (255, 100, 150), 13: (128, 0, 128), 14: (0, 128, 128),
        15: (128, 128, 128),
    }
    scale = 8
    h, w = len(grid), len(grid[0]) if grid else 0
    img = Image.new("RGB", (w * scale, h * scale))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            c = COLORS.get(grid[y][x], (128, 128, 128))
            for dy in range(scale):
                for dx in range(scale):
                    pixels[x * scale + dx, y * scale + dy] = c
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def ask_llm_for_direction(
    adapter: Any,
    grid: List[List[int]],
    life_number: int = 0,
    previous_deaths: List[str] = None,
    execution_feedback: str = "",
) -> Optional[str]:
    """Ask LLM to analyze the grid with VISION (image) + text.

    Renders grid as PNG image so multimodal LLM can SEE the game
    the same way a human does — forming concepts from visual patterns.
    Falls back to text-only if image fails.
    """
    from stg_engine.perception import _grid_to_text

    death_context = ""
    if previous_deaths:
        death_context = "\n".join(f"- {d}" for d in previous_deaths[-3:])
        death_context = f"\n\nPrevious deaths:\n{death_context}\nAvoid repeating these mistakes."

    feedback_block = ""
    if execution_feedback:
        feedback_block = f"\n\nPREVIOUS ATTEMPT RESULTS:\n{execution_feedback}\nUse this information to avoid repeating failed moves.\n"

    prompt_text = f"""You are playing a 2D puzzle game. Life #{life_number + 1}.{death_context}{feedback_block}

Look at the image carefully.

1. Identify the PLAYER (small unique sprite, often a cross/plus sign)
2. Identify OBJECTS (pushable boxes, switches, targets, doors)
3. Figure out the GAME MECHANICS (pushing? pattern matching? key+door?)
4. Figure out the GOAL (what needs to happen to win?)
5. Create a PLAN as a sequence of moves

IMPORTANT: Give me an ACTION SEQUENCE I can execute directly.
Use: U=up, D=down, L=left, R=right. Separate with commas.
Example: R,R,R,D,D,D,L,L,U means right 3, down 3, left 2, up 1.

Give ~20-50 moves that make progress toward the goal.
If you're unsure, explore toward the most promising area.

Answer format:
PLAYER: <what and where>
OBJECTS: <interactive objects>
MECHANICS: <game rules>
GOAL: <what to achieve>
ACTIONS: <comma-separated sequence like R,R,D,D,L,U,U,R>
REASON: <why this sequence>"""

    # Try vision (image) first, fall back to text
    image_b64 = _render_grid_to_base64(grid)

    if image_b64 and hasattr(adapter, 'complete'):
        # Send image directly via Gemini API (bypass text-only adapter)
        try:
            await adapter._ensure_auth()
            import httpx

            contents = [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    {"text": prompt_text},
                ]
            }]
            payload = {
                "model": adapter.config.model or "gemini-2.5-flash",
                "project": adapter._project_id,
                "request": {
                    "contents": contents,
                    "systemInstruction": {
                        "role": "user",
                        "parts": [{"text": "You are an expert game analyst with deep puzzle-solving skills."}]
                    },
                },
            }
            try:
                from skc.adapters.gemini_oauth import CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION
            except ImportError:
                raise NotImplementedError("Gemini OAuth integration requires the SKC package.")
            url = f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:generateContent"
            headers = {
                "Authorization": f"Bearer {adapter._access_token}",
                "Content-Type": "application/json",
                "User-Agent": f"GeminiCLI/0.1.0/{adapter.config.model} (win32; x64)",
            }

            import asyncio
            async with httpx.AsyncClient(timeout=300.0) as client:
                for attempt in range(3):
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 429:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    if resp.status_code == 401:
                        await adapter._refresh_access_token()
                        headers["Authorization"] = f"Bearer {adapter._access_token}"
                        continue
                    break

                if resp.status_code == 200:
                    data = resp.json()
                    response_data = data.get("response", data)
                    candidates = response_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        logger.debug(f"[LLM VISION] {text[:500]}")
                        return text
                else:
                    logger.debug(f"[LLM VISION HTTP] {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.debug(f"[LLM VISION FAIL] {e}, falling back to text")

    # Fallback: text-only
    try:
        from skc.types.llm import Message
    except ImportError:
        raise NotImplementedError("This LLM-dependent feature requires the SKC package. Use stg-engine for the core algorithms; SKC for LLM integration.")
    grid_text = _grid_to_text(grid)
    fallback_prompt = f"GRID (hex):\n{grid_text}\n\n{prompt_text}"
    messages = [Message(role="user", content=fallback_prompt)]
    try:
        response = await adapter.generate_response(
            messages,
            system_prompt="You are an expert game analyst.",
        )
        text = response.content.strip()
        logger.debug(f"[LLM DIRECTION TEXT] {text[:300]}")
        return text
    except Exception as e:
        logger.debug(f"[LLM DIRECTION FAIL] {e}")
        return None
