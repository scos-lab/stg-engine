"""STG Game Interface — STG's body for interacting with game environments.

Phase 12: Gives STG sensory input (vision) and motor output (actions).

STG is the subject. The game is the environment. This interface is the body.

Architecture:
    GameInterface
      ├── sense()    → get current frame from game
      ├── act()      → send action to game
      ├── observe()  → sense + extract reward signal
      └── run()      → autonomous loop: sense → think → act → observe → learn

Usage:
    interface = GameInterface(engine, adapter)
    interface.connect_arc(game_id="ls20")
    interface.run(max_steps=100)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from stg_engine.perception import (
    build_fixed_filters,
    grid_hash,
    init_learnable_filters,
    needs_llm_help,
    perceive_frame,
    perception_step,
    teach_from_llm,
)

logger = logging.getLogger(__name__)


class GameInterface:
    """STG's body — connects STG to a game environment.

    Provides:
        - Sensory input: reads game frames
        - Motor output: sends actions to game
        - Autonomous loop: STG drives the game, calling LLM only when needed
    """

    def __init__(
        self,
        engine: Any,  # STGEngine
        adapter: Any = None,  # LLMAdapter (optional, for teaching)
        explore_rate_start: float = 0.5,
        explore_rate_end: float = 0.1,
        explore_decay_steps: int = 50,
        teach_interval: int = 10,  # ask LLM every N new patterns
    ) -> None:
        self.engine = engine
        self.adapter = adapter

        # Exploration parameters
        self.explore_rate_start = explore_rate_start
        self.explore_rate_end = explore_rate_end
        self.explore_decay_steps = explore_decay_steps
        self.teach_interval = teach_interval

        # Ensure perception is initialized
        if not hasattr(engine, "_fixed_filters") or engine._fixed_filters is None:
            engine._fixed_filters = build_fixed_filters()
        if not hasattr(engine, "_perception_filters"):
            engine._perception_filters = init_learnable_filters()

        # Game state
        self._game_env = None
        self._game_id: Optional[str] = None
        self._arcade = None

        # I/O functions (pluggable)
        self._sense_fn: Optional[Callable] = None
        self._act_fn: Optional[Callable] = None
        self._reset_fn: Optional[Callable] = None
        self._is_done_fn: Optional[Callable] = None

        # Session stats
        self.step_count = 0
        self.llm_calls = 0
        self.levels_reached = 0
        self.new_patterns = 0
        self.history: List[Dict[str, Any]] = []

    # ═══════════════════════════════════════════════════════════
    # Connect to game environments
    # ═══════════════════════════════════════════════════════════

    def connect_arc(self, game_id: str, scorecard_tags: Optional[List[str]] = None):
        """Connect to ARC-AGI-3 game via API.

        Requires: arc_agi and arcengine packages.
        """
        from arc_agi import Arcade
        from arcengine import GameAction, GameState

        self._arcade = Arcade()
        tags = scorecard_tags or ["stg-autonomous"]
        card_id = self._arcade.open_scorecard(tags=tags)
        self._game_env = self._arcade.make(game_id, scorecard_id=card_id)
        self._game_id = game_id.split("-")[0]  # "ls20-xxx" → "ls20"

        # Wire I/O functions
        def sense():
            obs = self._game_env.observation_space
            if obs is None:
                return None, 0, [], False
            grid = [arr.tolist() for arr in obs.frame][-1] if hasattr(obs.frame[0], "tolist") else obs.frame[-1]
            level = obs.levels_completed
            available = [a for a in obs.available_actions if a != 0]
            done = obs.state == GameState.WIN
            return grid, level, available, done

        def act(action_id: int):
            action = GameAction.from_id(action_id)
            return self._game_env.step(action)

        def reset():
            obs = self._game_env.step(GameAction.RESET)
            grid = [arr.tolist() for arr in obs.frame][-1] if hasattr(obs.frame[0], "tolist") else obs.frame[-1]
            return grid, obs.levels_completed, [a for a in obs.available_actions if a != 0]

        def is_done():
            from arcengine import GameState
            obs = self._game_env.observation_space
            return obs is not None and obs.state == GameState.WIN

        self._sense_fn = sense
        self._act_fn = act
        self._reset_fn = reset
        self._is_done_fn = is_done

        logger.info(f"Connected to ARC game: {game_id}")

    def connect_custom(
        self,
        sense_fn: Callable,
        act_fn: Callable,
        reset_fn: Callable,
        is_done_fn: Callable,
        game_id: str = "custom",
    ):
        """Connect to any game environment via custom I/O functions.

        Args:
            sense_fn: () → (grid, level, available_actions, done)
            act_fn: (action_id) → observation
            reset_fn: () → (grid, level, available_actions)
            is_done_fn: () → bool
        """
        self._sense_fn = sense_fn
        self._act_fn = act_fn
        self._reset_fn = reset_fn
        self._is_done_fn = is_done_fn
        self._game_id = game_id
        logger.info(f"Connected to custom game: {game_id}")

    # ═══════════════════════════════════════════════════════════
    # Core I/O
    # ═══════════════════════════════════════════════════════════

    def sense(self) -> Tuple[Optional[List[List[int]]], int, List[int], bool]:
        """Read current game state. Returns (grid, level, available_actions, done)."""
        if self._sense_fn is None:
            raise RuntimeError("No game connected. Call connect_arc() or connect_custom() first.")
        return self._sense_fn()

    def act(self, action_id: int) -> Any:
        """Send action to game. Returns raw observation."""
        if self._act_fn is None:
            raise RuntimeError("No game connected.")
        return self._act_fn(action_id)

    def reset(self) -> Tuple[List[List[int]], int, List[int]]:
        """Reset game. Returns (grid, level, available_actions)."""
        if self._reset_fn is None:
            raise RuntimeError("No game connected.")
        return self._reset_fn()

    # ═══════════════════════════════════════════════════════════
    # Autonomous Loop
    # ═══════════════════════════════════════════════════════════

    def run(
        self,
        max_steps: int = 100,
        verbose: bool = True,
        on_step: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run the autonomous perception loop.

        STG drives: sense → think → act → observe → learn.
        Calls LLM only when encountering genuinely new patterns.

        Args:
            max_steps: maximum steps before stopping
            verbose: print step-by-step output
            on_step: optional callback(step_info) after each step

        Returns:
            Session summary dict
        """
        return asyncio.run(self._run_async(max_steps, verbose, on_step))

    async def _run_async(
        self,
        max_steps: int,
        verbose: bool,
        on_step: Optional[Callable],
    ) -> Dict[str, Any]:
        # Reset game
        grid, level, available = self.reset()
        self.step_count = 0
        self.llm_calls = 0
        self.levels_reached = level
        self.new_patterns = 0
        self.history = []

        prev_hash = None
        prev_action = None
        prev_level = level
        last_px_changed = -1

        action_names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACTION", 6: "CLICK"}

        if verbose:
            print(f"=== STG Autonomous Game: {self._game_id} ===\n")

        for step in range(max_steps):
            # Calculate explore rate (decay over time)
            progress = min(1.0, step / max(1, self.explore_decay_steps))
            explore_rate = self.explore_rate_start + (self.explore_rate_end - self.explore_rate_start) * progress

            # === SENSE ===
            grid, level, available, done = self.sense()
            if grid is None or done:
                if verbose:
                    print(f"  {'>>> GAME WON! <<<' if done else 'No frame available.'}")
                break

            # === THINK: Should I ask LLM for help? ===
            if (
                self.adapter is not None
                and needs_llm_help(self.engine, grid)
                and (self.new_patterns % self.teach_interval == 0 or self.step_count < 3)
            ):
                if verbose:
                    print(f"  [LLM] Teaching... ", end="", flush=True)
                try:
                    stl = await teach_from_llm(
                        self.engine, grid, self.adapter,
                        game_id=self._game_id,
                        context=f"Step {step}, level {level}",
                    )
                    self.llm_calls += 1
                    if verbose:
                        labels = len(stl)
                        print(f"{labels} labels learned")
                except Exception as e:
                    if verbose:
                        print(f"failed: {e}")

            # === THINK: Select action ===
            action_id, frame_hash, reason = perception_step(
                self.engine, grid, available,
                prev_hash=prev_hash, prev_action=prev_action,
                prev_level=prev_level, current_level=level,
                game_id=self._game_id, step_number=step,
                explore_rate=explore_rate,
                pixels_changed=last_px_changed,
            )

            # === ACT ===
            self.act(action_id)

            # === OBSERVE ===
            new_grid, new_level, new_available, done = self.sense()
            if new_grid is not None:
                px_changed = sum(
                    1 for y in range(len(grid)) for x in range(len(grid[0]))
                    if grid[y][x] != new_grid[y][x]
                )
            else:
                px_changed = 0

            # Track stats
            if new_level > level:
                self.levels_reached = new_level
            if frame_hash != prev_hash:
                self.new_patterns += 1

            step_info = {
                "step": step + 1,
                "action": action_names.get(action_id, "?"),
                "action_id": action_id,
                "px_changed": px_changed,
                "level": new_level,
                "reason": reason,
                "level_up": new_level > level,
                "frame_hash": frame_hash,
            }
            self.history.append(step_info)

            if verbose:
                marker = " <<<LEVEL UP>>>" if new_level > level else ""
                print(
                    f"{step+1:3d}: {step_info['action']:6s} {px_changed:3d}px "
                    f"lvl={new_level} {reason}{marker}"
                )

            if on_step:
                on_step(step_info)

            # Update for next step
            prev_hash = frame_hash
            prev_action = action_id
            prev_level = level
            last_px_changed = px_changed
            grid = new_grid if new_grid is not None else grid
            level = new_level
            available = new_available
            self.step_count = step + 1

            if done:
                if verbose:
                    print("  >>> GAME WON! <<<")
                break

        # Session summary
        summary = {
            "game_id": self._game_id,
            "steps": self.step_count,
            "levels_reached": self.levels_reached,
            "llm_calls": self.llm_calls,
            "new_patterns": self.new_patterns,
            "stg_nodes": len(self.engine._nodes),
            "stg_edges": len(self.engine._edges),
            "perception_frames": self.engine._perception_index.size
            if hasattr(self.engine, "_perception_index") and self.engine._perception_index
            else 0,
        }

        if verbose:
            print(f"\n=== Summary ===")
            print(f"Steps: {summary['steps']} | Levels: {summary['levels_reached']}")
            print(f"LLM calls: {summary['llm_calls']} | New patterns: {summary['new_patterns']}")
            print(f"STG: {summary['stg_nodes']} nodes, {summary['stg_edges']} edges, {summary['perception_frames']} frames")

        return summary

    # ═══════════════════════════════════════════════════════════
    # World Modeling Loop (Phase 13)
    # ═══════════════════════════════════════════════════════════

    def run_world_model(
        self,
        max_steps: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the world modeling loop (Phase 13).

        Unlike run() which uses random exploration + LLM labeling,
        this uses structured observation → rule inference → prediction →
        information-gain exploration.

        No LLM needed for basic rule discovery. LLM called only for
        semantic rules that can't be auto-inferred.
        """
        return asyncio.run(self._run_world_model_async(max_steps, verbose))

    async def _run_world_model_async(
        self,
        max_steps: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        from stg_engine.world_model import (
            world_modeling_step, analyze_change, infer_rules_from_observations,
            ACTION_NAMES, ChangeReport,
        )

        grid, level, available = self.reset()
        self.step_count = 0
        self.llm_calls = 0
        self.levels_reached = level
        self.history = []

        observations: List[ChangeReport] = []
        state_action_history: Dict[str, set] = {}
        prev_grid = None
        prev_action = None
        prev_level = level

        if verbose:
            print(f"=== STG World Modeling: {self._game_id} ===\n")

        for step in range(max_steps):
            grid_now, level_now, available, done = self.sense()
            if grid_now is None or done:
                if verbose and done:
                    print("  >>> GAME WON! <<<")
                break

            # World modeling step: observe → infer → predict → select
            action_id, reason, change_report = world_modeling_step(
                engine=self.engine,
                grid=grid_now,
                available_actions=available,
                prev_grid=prev_grid,
                prev_action=prev_action,
                prev_level=prev_level,
                # Note: adapter not passed here — LLM calls handled below
                current_level=level_now,
                observations=observations,
                state_action_history=state_action_history,
                adapter=self.adapter,
                step_number=step,
                game_id=self._game_id,
            )

            # LLM call if needed: initial, stuck, or anomaly
            # Cooldown: at least 5 steps between LLM calls (except initial)
            should_call_llm = False
            llm_trigger = ""
            if not hasattr(self, '_last_llm_step'):
                self._last_llm_step = -10  # allow initial calls
            if self.adapter is not None:
                if step < 2:
                    should_call_llm = True
                    llm_trigger = "initial"
                elif "anti_oscillation" in reason and (step - self._last_llm_step) >= 5:
                    should_call_llm = True
                    llm_trigger = "stuck"
                elif change_report and change_report.pixels_changed > 100 and (step - self._last_llm_step) >= 3:
                    should_call_llm = True
                    llm_trigger = "anomaly"

            if should_call_llm:
                from stg_engine.world_model import ask_llm_self_aware
                if verbose:
                    print(f"  [LLM] Asking ({llm_trigger})... ", end="", flush=True)
                try:
                    stl = await ask_llm_self_aware(
                        self.engine, self.adapter, observations,
                        prev_grid, grid_now, prev_action,
                        trigger_reason=llm_trigger,
                    )
                    self.llm_calls += 1
                    if verbose:
                        print(f"{len(stl)} rules learned")
                except Exception as e:
                    if verbose:
                        print(f"failed: {e}")
                finally:
                    self._last_llm_step = step  # cooldown even on failure

            # Execute
            self.act(action_id)

            # Sense result
            new_grid, new_level, new_available, done = self.sense()
            px_changed = 0
            if new_grid is not None:
                px_changed = sum(
                    1 for y in range(len(grid_now)) for x in range(len(grid_now[0]))
                    if grid_now[y][x] != new_grid[y][x]
                )

            if new_level > level_now:
                self.levels_reached = new_level

            action_name = ACTION_NAMES.get(action_id, "?")
            step_info = {
                "step": step + 1,
                "action": action_name,
                "action_id": action_id,
                "px_changed": px_changed,
                "level": new_level,
                "reason": reason,
                "level_up": new_level > level_now,
            }
            self.history.append(step_info)

            if verbose:
                marker = " <<<LEVEL UP>>>" if new_level > level_now else ""
                blocked = " [BLOCKED]" if px_changed < 10 and not step_info["level_up"] else ""
                print(
                    f"{step+1:3d}: {action_name:6s} {px_changed:3d}px "
                    f"lvl={new_level} {reason}{blocked}{marker}"
                )

            # Update for next step
            prev_grid = grid_now
            prev_action = action_id
            prev_level = level_now
            grid_now = new_grid if new_grid is not None else grid_now
            available = new_available
            self.step_count = step + 1

            if done:
                if verbose:
                    print("  >>> GAME WON! <<<")
                break

        # Final rule inference
        rules = infer_rules_from_observations(observations)

        # Summary
        n_blocked = sum(1 for o in observations if o.is_blocked)
        n_meaningful = sum(1 for o in observations if o.is_meaningful)
        unique_states = len(state_action_history)

        summary = {
            "game_id": self._game_id,
            "steps": self.step_count,
            "levels_reached": self.levels_reached,
            "observations": len(observations),
            "meaningful_moves": n_meaningful,
            "blocked_moves": n_blocked,
            "unique_states": unique_states,
            "rules_inferred": len(rules),
            "stg_nodes": len(self.engine._nodes),
            "stg_edges": len(self.engine._edges),
        }

        if verbose:
            print(f"\n=== World Model Summary ===")
            print(f"Steps: {summary['steps']} | Levels: {summary['levels_reached']}")
            print(f"Meaningful moves: {n_meaningful} | Blocked: {n_blocked} | Unique states: {unique_states}")
            print(f"Rules inferred: {len(rules)}")
            for r in rules:
                print(f"  {r[:120]}")
            print(f"STG: {summary['stg_nodes']} nodes, {summary['stg_edges']} edges")

        return summary

    # ═══════════════════════════════════════════════════════════
    # STG Brain Loop (Propagate-Select-Act-Learn)
    # ═══════════════════════════════════════════════════════════

    def run_stg_brain(
        self,
        max_steps: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the STG-brain game loop.

        STG IS the brain — no external planner.
        - Perceive: CNN features → find similar past frames
        - Think: propagate from current state → activation pattern = "thought"
        - Select: pick action with highest activation
        - Act: execute
        - Learn: Hebbian strengthen/weaken based on outcome
        - Remember: ingest (state, action, outcome) into STG
        """
        return asyncio.run(self._run_stg_brain_async(max_steps, verbose))

    async def _run_stg_brain_async(
        self,
        max_steps: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        from stg_engine.world_model import (
            analyze_change, ACTION_NAMES, ChangeReport,
            ask_llm_self_aware, infer_rules_from_observations,
        )
        from stg_engine.perception import grid_hash

        # Action name ↔ id mapping
        ACTION_IDS = {v: k for k, v in ACTION_NAMES.items()}
        MOVE_ACTIONS = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}

        grid, level, available = self.reset()
        self.step_count = 0
        self.llm_calls = 0
        self.levels_reached = level
        self.history = []
        self._last_llm_step = -10

        prev_grid = None
        prev_action = None
        prev_level = level
        prev_state_node = None
        observations: List[ChangeReport] = []
        consecutive_blocked = 0
        self._visited_states: Dict[str, int] = {}  # state_hash → visit count
        self._recent_blocked: List[int] = []  # recently blocked action ids

        if verbose:
            print(f"=== STG Brain: {self._game_id} ===\n")

        lives = 0  # how many times we've reset
        consecutive_zero = 0  # consecutive 0px steps (detect game over)
        llm_direction = None  # LLM's suggested direction (str like "RIGHT")
        death_notes: List[str] = []  # what we learned from each death
        need_direction = True  # ask LLM for direction at start of each life

        for step in range(max_steps):
            grid_now, level_now, available, done = self.sense()
            if grid_now is None or done:
                if verbose and done:
                    print("  >>> GAME WON! <<<")
                break

            # ── DETECT GAME OVER ──
            # If all actions return 0px for several steps, the game is over.
            # Reset and try again with accumulated knowledge.
            if consecutive_zero >= 3:
                lives += 1
                if verbose:
                    print(f"  ☠ GAME OVER (life #{lives}) — RESET with memory")
                    print(f"    STG: {len(self.engine._nodes)}n/{len(self.engine._edges)}e carried forward")
                # Record death in STG
                try:
                    self.engine.ingest_stl(
                        f'[{prev_state_node if prev_state_node else "Unknown"}] -> [GameOver_Life{lives}] '
                        f'::mod(rule="causal", confidence=0.99, '
                        f'description="energy depleted at step {step}")'
                    )
                except Exception:
                    pass
                # Reset game but KEEP STG intact
                try:
                    grid, level, available = self.reset()
                except Exception:
                    if verbose:
                        print("  Reset failed — ending session")
                    break
                prev_grid = None
                prev_action = None
                prev_level = level
                prev_state_node = None
                consecutive_zero = 0
                consecutive_blocked = 0
                self._recent_blocked.clear()
                self.levels_reached = level
                # Record death note and request new direction for next life
                death_notes.append(f"Life {lives}: died at step {step}, explored {len(self._visited_states)} states")
                need_direction = True  # Ask LLM again with updated context
                llm_direction = None
                continue

            # ── PERCEIVE ──
            # Hash current state and register in STG
            state_hash = grid_hash(grid_now)
            state_node = f"State_{state_hash}"

            # Ensure state node exists in STG
            if state_node not in self.engine._nodes:
                self.engine.add_node(state_node)

            # Track visits for novelty
            self._visited_states[state_hash] = self._visited_states.get(state_hash, 0) + 1
            visit_count = self._visited_states[state_hash]

            # ── LLM VISION (start of each life) ──
            # Ask LLM to analyze the grid semantically: what is what, where to go.
            # Then BRIDGE the semantic knowledge into the State/Action graph
            # so propagation can flow: Visual ↔ Semantic ↔ State ↔ Action
            if need_direction and self.adapter is not None:
                need_direction = False
                from stg_engine.world_model import ask_llm_for_direction
                if verbose:
                    print(f"  [LLM VISION] Analyzing grid... ", end="", flush=True)
                try:
                    analysis = await ask_llm_for_direction(
                        self.adapter, grid_now,
                        life_number=lives, previous_deaths=death_notes,
                    )
                    if analysis:
                        # Extract structured info
                        llm_player = llm_goal = llm_reason = ""
                        llm_mechanics = llm_plan = llm_objects = ""
                        for line in analysis.split("\n"):
                            up = line.strip().upper()
                            if up.startswith("DIRECTION:"):
                                llm_direction = line.split(":", 1)[1].strip().upper()
                            elif up.startswith("PLAYER:"):
                                llm_player = line.split(":", 1)[1].strip()[:150]
                            elif up.startswith("GOAL:"):
                                llm_goal = line.split(":", 1)[1].strip()[:200]
                            elif up.startswith("REASON:"):
                                llm_reason = line.split(":", 1)[1].strip()[:150]
                            elif up.startswith("MECHANICS:"):
                                llm_mechanics = line.split(":", 1)[1].strip()[:200]
                            elif up.startswith("PLAN:"):
                                llm_plan = line.split(":", 1)[1].strip()[:200]
                            elif up.startswith("OBJECTS:"):
                                llm_objects = line.split(":", 1)[1].strip()[:200]

                        # ── BRIDGE: connect semantic knowledge to current state ──
                        # This creates edges between the three "islands":
                        # State ↔ Semantic ↔ Action direction

                        # 1. Current state → LLM semantic understanding
                        try:
                            self.engine.ingest_stl(
                                f'[{state_node}] -> [Semantic_Life{lives}] '
                                f'::mod(rule="definitional", confidence=0.9, '
                                f'description="player={llm_player}")'
                            )
                        except Exception:
                            pass

                        # 2. Semantic → Goal (persistent across steps)
                        if llm_goal:
                            try:
                                self.engine.ingest_stl(
                                    f'[Semantic_Life{lives}] -> [Goal_Current] '
                                    f'::mod(rule="causal", confidence=0.85, '
                                    f'description="{llm_goal}")'
                                )
                            except Exception:
                                pass

                        # 3. Goal → Suggested direction action (the KEY bridge)
                        if llm_direction:
                            try:
                                self.engine.ingest_stl(
                                    f'[Goal_Current] -> [Action_{llm_direction}] '
                                    f'::mod(rule="causal", confidence=0.8, '
                                    f'description="{llm_reason}")'
                                )
                            except Exception:
                                pass

                        # 4. Visual frame → Semantic (bridge vision to meaning)
                        visual_node = f"Visual:frame_{state_hash}"
                        if visual_node in self.engine._nodes:
                            try:
                                self.engine.ingest_stl(
                                    f'[{visual_node}] -> [Semantic_Life{lives}] '
                                    f'::mod(rule="definitional", confidence=0.7, '
                                    f'description="visual-semantic bridge")'
                                )
                            except Exception:
                                pass

                        # 5. Mechanics and plan → STG knowledge
                        if llm_mechanics:
                            try:
                                self.engine.ingest_stl(
                                    f'[Semantic_Life{lives}] -> [Mechanics_Current] '
                                    f'::mod(rule="definitional", confidence=0.85, '
                                    f'description="{llm_mechanics}")'
                                )
                            except Exception:
                                pass
                        if llm_plan:
                            try:
                                self.engine.ingest_stl(
                                    f'[Goal_Current] -> [Plan_Life{lives}] '
                                    f'::mod(rule="causal", confidence=0.8, '
                                    f'description="{llm_plan}")'
                                )
                            except Exception:
                                pass

                        if verbose:
                            dir_str = llm_direction or "?"
                            print(f"→ {dir_str}")
                            for line in analysis.split("\n"):
                                if any(line.strip().upper().startswith(k)
                                       for k in ("PLAYER:", "OBJECTS:", "MECHANICS:",
                                                  "GOAL:", "PLAN:", "DIRECTION:", "REASON:")):
                                    print(f"    {line.strip()}")
                    self.llm_calls += 1
                except Exception as e:
                    if verbose:
                        print(f"failed: {e}")
                finally:
                    self._last_llm_step = step

            # ── REMEMBER (ingest previous step's outcome) ──
            if prev_grid is not None and prev_action is not None:
                change = analyze_change(
                    prev_grid, grid_now, prev_action, prev_level, level_now
                )
                observations.append(change)

                action_name = ACTION_NAMES.get(prev_action, f"A{prev_action}")
                action_node = f"Action_{action_name}"
                outcome_node = state_node  # outcome = next state

                # ── SURPRISE (Δρ anomaly detection) ──
                # Compute expected change from historical observations
                same_action_obs = [o for o in observations[:-1]
                                   if o.action_id == prev_action]
                if same_action_obs:
                    expected_px = sum(o.pixels_changed for o in same_action_obs) / len(same_action_obs)
                else:
                    expected_px = 30.0  # neutral prior
                surprise = abs(change.pixels_changed - expected_px) / max(expected_px, 1.0)

                # Ingest: State → Action edge
                try:
                    self.engine.ingest_stl(
                        f'[{prev_state_node}] -> [{action_node}] '
                        f'::mod(rule="causal", confidence=0.8, '
                        f'description="tried {action_name} from state {prev_state_node}")'
                    )
                except Exception:
                    pass

                # ── LEARN ──
                if change.level_up:
                    # Phase transition! Maximum surprise.
                    try:
                        self.engine.ingest_stl(
                            f'[{action_node}] -> [{outcome_node}] '
                            f'::mod(rule="causal", confidence=0.99, '
                            f'description="level_up")'
                        )
                    except Exception:
                        pass
                    path = [prev_state_node, action_node, outcome_node]
                    self.engine.learn_from_path(path, strength=1.0)
                    try:
                        self.engine.ingest_stl(
                            f'[{outcome_node}] -> [Reward_LevelUp] '
                            f'::mod(rule="causal", confidence=0.99, '
                            f'strength=1.0, description="reached level {level_now}")'
                        )
                    except Exception:
                        pass
                    consecutive_blocked = 0
                    self._recent_blocked.clear()

                elif change.is_blocked:
                    # Record that this action was tried and led to a dead end.
                    # This prevents novelty scorer from treating untried-but-blocked
                    # directions as "unknown" (they are known-bad).
                    blocked_node = f"Blocked_{prev_state_node}"
                    try:
                        self.engine.ingest_stl(
                            f'[{action_node}] -> [{blocked_node}] '
                            f'::mod(rule="causal", confidence=0.95, '
                            f'description="blocked from {prev_state_node}")'
                        )
                    except Exception:
                        pass
                    # Mark blocked outcome as "visited many times" → kills novelty
                    blocked_hash = blocked_node.split("_", 1)[1] if "_" in blocked_node else blocked_node
                    self._visited_states[blocked_hash] = 999

                    # Weaken action edges
                    for edge in self.engine._edges:
                        if edge.target == action_node:
                            edge.salience = max(0.05, edge.salience * 0.7)
                        if edge.source == action_node:
                            edge.salience = max(0.05, edge.salience * 0.7)
                    consecutive_blocked += 1
                    # Track recently blocked directions
                    if prev_action not in self._recent_blocked:
                        self._recent_blocked.append(prev_action)

                elif change.is_meaningful:
                    # Ingest: Action → new State
                    try:
                        self.engine.ingest_stl(
                            f'[{action_node}] -> [{outcome_node}] '
                            f'::mod(rule="causal", confidence=0.85, '
                            f'description="moved_{change.pixels_changed}px")'
                        )
                    except Exception:
                        pass

                    # Novelty bonus: new state = stronger reinforce
                    is_novel = self._visited_states.get(state_hash, 0) <= 1

                    # ── SURPRISE-DRIVEN REINFORCEMENT ──
                    if surprise > 1.0 or is_novel:
                        # High anomaly or novel state: ρ field in uncharted territory
                        # Connect to Surprise hub with strength proportional to surprise + novelty
                        novelty_bonus = 0.3 if is_novel else 0.0
                        surprise_strength = min((surprise / 3.0) + novelty_bonus, 1.0)
                        try:
                            self.engine.ingest_stl(
                                f'[{outcome_node}] -> [Surprise_Hub] '
                                f'::mod(rule="causal", confidence={0.7 + surprise_strength * 0.25:.2f}, '
                                f'description="anomaly_surprise={surprise:.2f}_px={change.pixels_changed}")'
                            )
                        except Exception:
                            pass
                        # Strong reinforce: this path produced anomalous change
                        path = [prev_state_node, action_node, outcome_node]
                        self.engine.learn_from_path(path, strength=surprise_strength)

                        if verbose:
                            print(f"  ★ SURPRISE={surprise:.1f} "
                                  f"(expected={expected_px:.0f}, got={change.pixels_changed})")
                    else:
                        # Normal movement — mild reinforce
                        path = [prev_state_node, action_node, outcome_node]
                        self.engine.learn_from_path(path, strength=0.3)

                    consecutive_blocked = 0
                    self._recent_blocked.clear()
                else:
                    consecutive_blocked += 1

            # ── THINK (propagate) ──
            # Seed activation directly from current state node and propagate
            # through connected Action/Outcome/Reward nodes.
            # This is STG "thinking" — activation flows through learned paths.
            # Current state activation inversely proportional to visit count
            # Novel states get full activation; revisited states are dampened
            novelty = 1.0 / (1.0 + (visit_count - 1) * 0.3)
            seed = {state_node: novelty}
            # Seed states connected to Surprise/Reward hubs.
            # Since propagation flows forward (successors), we seed the
            # PREDECESSORS of these hubs — states that led to anomalies.
            # This way activation flows: surprise_state → Action → next outcomes
            for hub_name, hub_weight in [("Surprise_Hub", 0.4), ("Reward_LevelUp", 0.8)]:
                if hub_name not in self.engine._nodes:
                    continue
                # Find predecessors (states that connect TO this hub)
                for edge in self.engine._edges:
                    if edge.target == hub_name:
                        seed[edge.source] = seed.get(edge.source, 0) + hub_weight * edge.salience
            # Also seed from visually similar past states
            try:
                from stg_engine.perception import find_similar_states, perceive_frame
                perceive_frame(self.engine, grid_now, self._game_id, step, level_now)
                similar = find_similar_states(self.engine, grid_now, top_k=3)
                for sim_name, sim_score in similar:
                    if not sim_name.endswith(state_hash):  # exclude self
                        seed[sim_name] = sim_score * 0.5
                        # Also seed the State: node linked to this visual frame
                        for edge in self.engine._edges:
                            if edge.source == sim_name and edge.target.startswith("State_"):
                                seed[edge.target] = sim_score * 0.7
                            if edge.target == sim_name and edge.source.startswith("State_"):
                                seed[edge.source] = sim_score * 0.7
            except Exception:
                pass  # perception not critical

            activated = self.engine._propagate_from_seeds(
                activation_map=seed,
                decay=0.65,
                iterations=5,
                threshold=0.05,
                normalize=True,
                input_text=state_node,
                token_count=1,
                seed_count=len(seed),
            )
            metrics = self.engine.last_propagation_metrics

            # ── SELECT (pick action from activation pattern) ──
            action_id = self._select_action_from_activation(
                metrics, available, MOVE_ACTIONS, step, consecutive_blocked,
                self._recent_blocked, llm_direction,
            )
            # Debug: show propagation details
            if verbose and step < 8:
                print(f"    seeds={list(seed.keys())[:3]} activated={len(activated)} nodes")
                for aid, aname in MOVE_ACTIONS.items():
                    node = self.engine._nodes.get(f"Action_{aname}")
                    act_val = node.activation if node else 0
                    if act_val > 0:
                        print(f"    Action_{aname}={act_val:.4f}", end=" ")
                if any(self.engine._nodes.get(f"Action_{n}") and
                       self.engine._nodes[f"Action_{n}"].activation > 0
                       for n in MOVE_ACTIONS.values()):
                    print()  # newline after action activations

            # ── LLM (only when truly stuck or initial) ──
            if self.adapter is not None:
                should_call_llm = False
                llm_trigger = ""
                if step < 2:
                    should_call_llm = True
                    llm_trigger = "initial"
                elif consecutive_blocked >= 4 and (step - self._last_llm_step) >= 5:
                    should_call_llm = True
                    llm_trigger = "stuck"

                if should_call_llm:
                    if verbose:
                        print(f"  [LLM] Asking ({llm_trigger})... ", end="", flush=True)
                    try:
                        stl = await ask_llm_self_aware(
                            self.engine, self.adapter, observations,
                            prev_grid, grid_now, prev_action,
                            trigger_reason=llm_trigger,
                        )
                        self.llm_calls += 1
                        if verbose:
                            print(f"{len(stl)} rules learned")
                    except Exception as e:
                        if verbose:
                            print(f"failed: {e}")
                    finally:
                        self._last_llm_step = step

            # ── ACT ──
            self.act(action_id)

            # Sense result
            new_grid, new_level, new_available, done = self.sense()
            px_changed = 0
            if new_grid is not None:
                px_changed = sum(
                    1 for y in range(len(grid_now)) for x in range(len(grid_now[0]))
                    if grid_now[y][x] != new_grid[y][x]
                )

            if new_level > level_now:
                self.levels_reached = new_level

            # Track consecutive zero-change steps (game over detection)
            if px_changed == 0:
                consecutive_zero += 1
            else:
                consecutive_zero = 0

            action_name = ACTION_NAMES.get(action_id, "?")
            blocked = px_changed < 10 and new_level <= level_now
            level_up = new_level > level_now

            step_info = {
                "step": step + 1, "action": action_name,
                "action_id": action_id, "px_changed": px_changed,
                "level": new_level, "level_up": level_up,
            }
            self.history.append(step_info)

            if verbose:
                marker = " <<<LEVEL UP>>>" if level_up else ""
                blk = " [BLOCKED]" if blocked else ""
                n_nodes = len(self.engine._nodes)
                n_edges = len(self.engine._edges)
                print(
                    f"{step+1:3d}: {action_name:6s} {px_changed:3d}px "
                    f"lvl={new_level} STG={n_nodes}n/{n_edges}e{blk}{marker}"
                )

            # Update for next step
            prev_grid = grid_now
            prev_action = action_id
            prev_level = level_now
            prev_state_node = state_node
            grid_now = new_grid if new_grid is not None else grid_now
            available = new_available
            self.step_count = step + 1

            if done:
                if verbose:
                    print("  >>> GAME WON! <<<")
                break

        # Summary
        n_blocked = sum(1 for s in self.history if s.get("px_changed", 0) < 10 and not s.get("level_up"))
        n_meaningful = sum(1 for s in self.history if s.get("px_changed", 0) >= 10 or s.get("level_up"))

        summary = {
            "game_id": self._game_id,
            "steps": self.step_count,
            "levels_reached": self.levels_reached,
            "meaningful_moves": n_meaningful,
            "blocked_moves": n_blocked,
            "stg_nodes": len(self.engine._nodes),
            "stg_edges": len(self.engine._edges),
            "llm_calls": self.llm_calls,
            "lives": lives,
        }

        if verbose:
            print(f"\n=== STG Brain Summary ===")
            print(f"Steps: {summary['steps']} | Levels: {summary['levels_reached']} | Lives: {lives}")
            print(f"Meaningful: {n_meaningful} | Blocked: {n_blocked}")
            print(f"STG: {summary['stg_nodes']} nodes, {summary['stg_edges']} edges")
            print(f"LLM calls: {self.llm_calls}")

        return summary

    # ═══════════════════════════════════════════════════════════
    # Visual Closed-Loop (See → Plan → Execute → See → Replan)
    # ═══════════════════════════════════════════════════════════

    def run_visual_loop(
        self,
        max_steps: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Visual closed-loop: LLM sees screenshot → plans action sequence → execute → replan.

        Architecture:
            render screenshot → LLM vision → action sequence (R,R,D,D,L...)
            → execute step by step → detect surprise/block → replan
        """
        return asyncio.run(self._run_visual_loop_async(max_steps, verbose))

    async def _run_visual_loop_async(
        self,
        max_steps: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        from stg_engine.world_model import (
            analyze_change, ACTION_NAMES, ask_llm_for_direction,
        )
        from stg_engine.perception import grid_hash

        ACTION_MAP = {"U": 1, "D": 2, "L": 3, "R": 4,
                      "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
        ACTION_NAMES_SHORT = {1: "U", 2: "D", 3: "L", 4: "R"}

        grid, level, available = self.reset()
        self.step_count = 0
        self.llm_calls = 0
        self.levels_reached = level
        self.history = []
        lives = 0
        death_notes: List[str] = []
        consecutive_zero = 0

        if verbose:
            print(f"=== Visual Loop: {self._game_id} ===\n")

        step = 0
        while step < max_steps:
            # ── SEE: render current frame → LLM ──
            grid_now, level_now, available, done = self.sense()
            if grid_now is None or done:
                if verbose and done:
                    print("  >>> GAME WON! <<<")
                break

            if verbose:
                print(f"\n  [VISION #{self.llm_calls + 1}] Rendering screenshot for LLM...")

            # Build feedback from last execution
            if not hasattr(self, '_last_feedback'):
                self._last_feedback = ""

            analysis = None
            try:
                analysis = await ask_llm_for_direction(
                    self.adapter, grid_now,
                    life_number=lives, previous_deaths=death_notes,
                    execution_feedback=self._last_feedback,
                )
                self.llm_calls += 1
            except Exception as e:
                if verbose:
                    print(f"  LLM failed: {e}")

            # ── PLAN: extract action sequence ──
            action_sequence = []
            if analysis:
                # Show analysis
                if verbose:
                    for line in analysis.split("\n"):
                        up = line.strip().upper()
                        if any(up.startswith(k) for k in
                               ("PLAYER:", "OBJECTS:", "MECHANICS:", "GOAL:",
                                "ACTIONS:", "REASON:", "PLAN:")):
                            print(f"    {line.strip()}")

                # Parse ACTIONS line
                for line in analysis.split("\n"):
                    if line.strip().upper().startswith("ACTIONS:"):
                        actions_str = line.split(":", 1)[1].strip()
                        # Parse comma-separated: R,R,D,D,L,U
                        for token in actions_str.replace(" ", "").split(","):
                            token = token.strip().upper()
                            if token in ACTION_MAP:
                                action_sequence.append(ACTION_MAP[token])

            if not action_sequence:
                # Fallback: explore randomly for a few steps
                import random
                candidates = [a for a in available if a in (1, 2, 3, 4)]
                action_sequence = [random.choice(candidates) for _ in range(10)] if candidates else [1] * 5

            if verbose:
                seq_str = ",".join(ACTION_NAMES_SHORT.get(a, "?") for a in action_sequence)
                print(f"  Plan ({len(action_sequence)} moves): {seq_str}")

            # ── EXECUTE: run action sequence ──
            prev_grid = grid_now
            replan_reason = None
            self._seq_blocked = 0
            feedback_lines = []  # track what happened for next LLM call

            for i, action_id in enumerate(action_sequence):
                if step >= max_steps:
                    break

                self.act(action_id)
                new_grid, new_level, new_available, done = self.sense()

                # Compute change
                px_changed = 0
                if new_grid is not None:
                    px_changed = sum(
                        1 for y in range(len(prev_grid)) for x in range(len(prev_grid[0]))
                        if prev_grid[y][x] != new_grid[y][x]
                    )

                action_name = ACTION_NAMES_SHORT.get(action_id, "?")
                blocked = px_changed < 10 and new_level <= level_now
                level_up = new_level > level_now

                if level_up:
                    self.levels_reached = new_level

                # Collect execution feedback
                status = "BLOCKED" if blocked else ("LEVEL_UP!" if level_up else f"ok({px_changed}px)")
                feedback_lines.append(f"{action_name}={status}")

                step += 1
                self.step_count = step

                step_info = {
                    "step": step, "action": action_name,
                    "px_changed": px_changed, "level": new_level,
                    "level_up": level_up,
                }
                self.history.append(step_info)

                if verbose:
                    marker = " <<<LEVEL UP>>>" if level_up else ""
                    blk = " [X]" if blocked else ""
                    print(f"  {step:3d}: {action_name} {px_changed:3d}px{blk}{marker}", end="")
                    # Print on same line, newline every 5 steps
                    if (i + 1) % 5 == 0 or blocked or level_up:
                        print()
                    else:
                        print("  ", end="")

                # ── CHECK: should we replan? ──
                if done:
                    if verbose:
                        print("\n  >>> GAME WON! <<<")
                    replan_reason = "won"
                    break

                if level_up:
                    replan_reason = "level_up"
                    if verbose:
                        print()
                    break

                if px_changed == 0:
                    consecutive_zero += 1
                else:
                    consecutive_zero = 0

                # Game over detection
                if consecutive_zero >= 3:
                    lives += 1
                    replan_reason = "game_over"
                    death_notes.append(
                        f"Life {lives}: died at step {step}")
                    if verbose:
                        print(f"\n  ☠ GAME OVER (life #{lives}) — RESET with memory")
                    try:
                        grid, level, available = self.reset()
                    except Exception:
                        break
                    consecutive_zero = 0
                    level_now = level
                    break

                # Large surprise → replan (something unexpected happened)
                if px_changed > 100:
                    replan_reason = f"surprise_{px_changed}px"
                    if verbose:
                        print(f"\n  ★ BIG CHANGE ({px_changed}px) — replanning")
                    break

                # Blocked → skip this move, continue with rest of plan
                # Only replan if blocked too many times in the same sequence
                if blocked:
                    if not hasattr(self, '_seq_blocked'):
                        self._seq_blocked = 0
                    self._seq_blocked += 1
                    if self._seq_blocked >= 5:
                        # Stuck — replan with new vision
                        replan_reason = f"stuck_{self._seq_blocked}_blocks"
                        self._seq_blocked = 0
                        if verbose:
                            print()
                        break
                    # Otherwise just skip this move, continue sequence
                    continue

                prev_grid = new_grid if new_grid is not None else prev_grid
                available = new_available
                level_now = new_level

                if done:
                    break

            # End of action sequence or replan triggered
            # Save feedback for next LLM call
            if feedback_lines:
                self._last_feedback = (
                    f"Plan was: {','.join(ACTION_NAMES_SHORT.get(a,'?') for a in action_sequence)}\n"
                    f"Results: {' | '.join(feedback_lines)}\n"
                    f"Replan reason: {replan_reason or 'sequence completed'}"
                )

            if verbose and not replan_reason:
                print()  # newline after sequence

            if replan_reason == "won":
                break

            # Refresh grid for next vision call
            grid_now, level_now, available, done = self.sense()
            if grid_now is not None:
                prev_grid = grid_now

        # Summary
        n_blocked = sum(1 for s in self.history if s.get("px_changed", 0) < 10 and not s.get("level_up"))
        n_meaningful = sum(1 for s in self.history if s.get("px_changed", 0) >= 10 or s.get("level_up"))

        summary = {
            "game_id": self._game_id,
            "steps": self.step_count,
            "levels_reached": self.levels_reached,
            "meaningful_moves": n_meaningful,
            "blocked_moves": n_blocked,
            "stg_nodes": len(self.engine._nodes),
            "stg_edges": len(self.engine._edges),
            "llm_calls": self.llm_calls,
            "lives": lives,
        }

        if verbose:
            print(f"\n=== Visual Loop Summary ===")
            print(f"Steps: {summary['steps']} | Levels: {summary['levels_reached']} | Lives: {lives}")
            print(f"Meaningful: {n_meaningful} | Blocked: {n_blocked}")
            print(f"LLM vision calls: {self.llm_calls}")

        return summary

    def _select_action_from_activation(
        self,
        metrics,
        available: List[int],
        move_actions: Dict[int, str],
        step: int,
        consecutive_blocked: int,
        recent_blocked: List[int] = None,
        llm_direction: Optional[str] = None,
    ) -> int:
        """Select action that leads toward UNKNOWN territory.

        Core principle from density monism:
        The agent seeks regions of ρ it hasn't measured yet.
        Known territory has low value. Unknown territory has high value.

        Score = novelty_score + surprise_score + activation_score
        - novelty: does this action lead to a never-visited state?
        - surprise: has this action produced anomalous Δρ before?
        - activation: STG propagation signal (Hebbian memory)
        """
        import random

        blocked_set = set(recent_blocked) if recent_blocked else set()
        candidates = [a for a in available if a in move_actions
                      and a != 0 and a not in blocked_set]
        if not candidates:
            # All directions blocked — clear and try any
            candidates = [a for a in available if a in move_actions and a != 0]
        if not candidates:
            return available[0] if available else 1

        action_scores: Dict[int, float] = {}

        for aid in candidates:
            aname = move_actions[aid]
            action_node = f"Action_{aname}"
            score = 0.0

            # ── NOVELTY: where does this action lead? ──
            # Look up known transitions from current state via this action
            outcome_visits = []
            has_blocked_outcome = False
            for edge in self.engine._edges:
                if edge.source == action_node:
                    if edge.target.startswith("State_"):
                        target_hash = edge.target.split("_", 1)[1]
                        v = self._visited_states.get(target_hash, 0)
                        outcome_visits.append(v)
                    elif edge.target.startswith("Blocked_"):
                        has_blocked_outcome = True

            if has_blocked_outcome and not outcome_visits:
                # Only known outcome is blocked → low score
                score += 0.1
            elif not outcome_visits:
                # Never tried this action → unknown outcome → high novelty
                score += 3.0
            else:
                min_visits = min(outcome_visits)
                if min_visits == 0:
                    # Known to lead to unvisited state → high novelty
                    score += 2.0
                else:
                    # Known outcome, visited → score inversely proportional
                    score += 1.0 / (1.0 + min_visits)

            # ── SURPRISE: has this action produced anomalies? ──
            # Check if action connects to Surprise_Hub
            for edge in self.engine._edges:
                if edge.source == action_node and "Surprise" in edge.target:
                    score += 1.0 * edge.salience

            # Also check indirect: action → state → Surprise_Hub
            for edge in self.engine._edges:
                if edge.source == action_node and edge.target.startswith("State_"):
                    for e2 in self.engine._edges:
                        if e2.source == edge.target and e2.target == "Surprise_Hub":
                            score += 0.5 * e2.salience

            # ── ACTIVATION: STG propagation memory ──
            node = self.engine._nodes.get(action_node)
            if node and node.activation > 0.01:
                score += node.activation * 0.5  # damped — novelty dominates

            # ── REWARD: connections to level-up ──
            for edge in self.engine._edges:
                if edge.source == action_node:
                    for e2 in self.engine._edges:
                        if e2.source == edge.target and e2.target == "Reward_LevelUp":
                            score += 5.0 * e2.salience  # very strong pull

            # ── LLM DIRECTION: semantic guidance from grid analysis ──
            if llm_direction and aname.upper() == llm_direction:
                score += 1.5  # LLM says go this way

            action_scores[aid] = score

        # Select: mostly greedy on score, small exploration chance
        best = max(action_scores, key=action_scores.get)
        explore_rate = max(0.05, 0.2 - step * 0.003)
        if random.random() < explore_rate:
            return random.choice(candidates)
        return best
