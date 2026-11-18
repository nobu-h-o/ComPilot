"""Random-search baseline (no LLM).

Explores the same transformation space as ComPilot (Fuse, Distribute,
Interchange, Parallelize, Tile2D, Tile3D, Unroll, Skew, Reverse) and uses the same compiler
backend, feedback cache and result bookkeeping, but picks transformations
uniformly at random instead of asking an LLM. It exists to measure how much the
LLM adds over undirected search under the same budget.

Search procedure (a random walk over schedules):
  - The walk starts from the unoptimized program (empty schedule).
  - At each step, every transformation that can be appended to the current
    schedule is enumerated, one is drawn at random, and the extended schedule is
    sent to Tiramisu.
  - If Tiramisu reports it as illegal (or it is invalid / crashes), it is dropped:
    it is never drawn again and the walk stays where it was. Only legal schedules
    are executed and become the base for further extensions.
  - The walk restarts from the empty schedule with probability `restart_prob`
    after each executed schedule, or when a schedule reaches `max_schedule_length`
    transformations. When the current schedule has no untried extensions left, it
    jumps to a random legal schedule found so far that still has some.
  - Unrolling is always the last transformation of a schedule (a backend rule).

The search stops after `nb_schedules` schedules were executed, after
`max_attempts` schedules were proposed (whatever their outcome), or when no
legal schedule found so far has untried extensions left.
"""
import random
import re
import traceback

from compilot_utils import (
    llm_sched_str_to_sched, get_compiler_response, get_innermost_level, logger,
)

ALL_TRANSFORMATIONS = ("parallelize", "interchange", "reverse", "skew",
                       "tile2d", "tile3d", "unroll", "fuse", "distribute")

DEFAULTS = {
    "seed": 0,
    "nb_schedules": None,           # None -> use max_llm_responses from the main config
    "max_attempts": None,           # None -> 20 * nb_schedules
    "max_schedule_length": 5,
    "restart_prob": 0.2,
    "transformations": list(ALL_TRANSFORMATIONS),
    "tiling_factors": [16, 32, 64, 128],
    "unroll_factors": [2, 4, 8, 16],
}


def resolve_random_config(random_cfg, max_llm_responses):
    """Merge the user's `random_search` block over the defaults."""
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (random_cfg or {}).items() if v is not None})
    if cfg["nb_schedules"] is None:
        cfg["nb_schedules"] = max_llm_responses
    if cfg["max_attempts"] is None:
        cfg["max_attempts"] = 20 * cfg["nb_schedules"]
    unknown = set(t.lower() for t in cfg["transformations"]) - set(ALL_TRANSFORMATIONS)
    if unknown:
        raise ValueError(f"random_search.transformations: unknown {sorted(unknown)}; "
                         f"supported: {list(ALL_TRANSFORMATIONS)}")
    cfg["transformations"] = [t.lower() for t in cfg["transformations"]]
    return cfg


def _join(prefix, transf):
    return f"{prefix}+{transf}" if prefix else transf


def _candidate_transformations(schedule_str, schedule, compute_blocks_IDs, cfg):
    """All single transformations that could be appended to `schedule_str`, using the
    loop depths of the current (already transformed) schedule tree. Candidates that
    turn out to be structurally invalid are filtered out later by the parser."""
    allowed = set(cfg["transformations"])
    applied = schedule_str.split('+') if schedule_str else []
    unrolled = {t.split('.')[0] for t in applied if '.unroll(' in t.lower()}
    fused = set()  # unordered comp_ID pairs already fused (re-fusing them is a no-op)
    for t in applied:
        m = re.match(r'(\w+)\.fuse\((\w+),', t, re.IGNORECASE)
        if m:
            fused.add(frozenset(m.groups()))

    depth = {}
    for comp_id, comps in compute_blocks_IDs.items():
        try:
            depth[comp_id] = get_innermost_level(comps[0], schedule)
        except Exception:
            continue

    cands = []
    for comp_id, d in depth.items():
        if comp_id in unrolled:
            continue  # unrolling must stay the last transformation on a computation
        levels = range(d + 1)
        if "parallelize" in allowed:
            cands += [f"{comp_id}.Parallelize(L{i})" for i in levels if i <= 1]
        if "reverse" in allowed:
            cands += [f"{comp_id}.Reverse(L{i})" for i in levels]
        if "interchange" in allowed:
            cands += [f"{comp_id}.Interchange(L{i},L{j})" for i in levels for j in levels if i < j]
        if "skew" in allowed:
            cands += [f"{comp_id}.Skew(L{i},L{i+1})" for i in range(d)]
        if "tile2d" in allowed:
            cands += [f"{comp_id}.Tile2D(L{i},L{i+1},{f1},{f2})"
                      for i in range(d) for f1 in cfg["tiling_factors"] for f2 in cfg["tiling_factors"]]
        if "tile3d" in allowed:
            cands += [f"{comp_id}.Tile3D(L{i},L{i+1},L{i+2},{f1},{f2},{f3})"
                      for i in range(d - 1) for f1 in cfg["tiling_factors"]
                      for f2 in cfg["tiling_factors"] for f3 in cfg["tiling_factors"]]
        if "distribute" in allowed:
            cands += [f"{comp_id}.Distribute(L{i})" for i in levels]
        if "unroll" in allowed:
            cands += [f"{comp_id}.Unroll(L{d},{f})" for f in cfg["unroll_factors"]]
        if "fuse" in allowed:
            for other_id, od in depth.items():
                if (other_id != comp_id and other_id not in unrolled
                        and frozenset((comp_id, other_id)) not in fused):
                    cands += [f"{comp_id}.Fuse({other_id},L{i})" for i in range(min(d, od) + 1)]
    return cands


def _sample_by_kind(rng, cands):
    """Draw a transformation type uniformly first, then an instance of it, so that
    types with many parameterizations (tiling) do not crowd out the others."""
    by_kind = {}
    for c in cands:
        by_kind.setdefault(c.split('.')[1].split('(')[0].lower(), []).append(c)
    kind = rng.choice(sorted(by_kind))
    return rng.choice(by_kind[kind])


def run_random_search(tiramisu_program, annots, compute_blocks_IDs, initial_exec_time,
                      explored_scheds, cfg, seed, min_runs, max_runs, time_budget):
    """Run the random walk for one program. Appends every schedule sent to the
    compiler to `explored_scheds` (same 5-tuple format as the LLM driver) and
    returns a list of log lines describing the search."""
    rng = random.Random(seed)
    log = []
    tried = set()          # every schedule string already drawn (never redrawn)
    tree_cache = {}        # schedule_str -> built Schedule (for candidate enumeration)
    executed = 0
    attempts = 0
    current = ''
    executed_list = []     # legal executed schedules, in order (bases the walk can jump back to)
    dead = set()           # schedules with no untried extensions left

    def build(sched_str):
        if sched_str not in tree_cache:
            tree_cache[sched_str] = (llm_sched_str_to_sched(sched_str, tiramisu_program, annots, compute_blocks_IDs)
                                     if sched_str else None)
        return tree_cache[sched_str]

    from tiralib.tiramisu import Schedule
    base_schedule = Schedule(tiramisu_program)

    while executed < cfg["nb_schedules"] and attempts < cfg["max_attempts"]:
        if current and len(current.split('+')) >= cfg["max_schedule_length"]:
            current = ''
        sched_obj = build(current) if current else base_schedule
        cands = [c for c in _candidate_transformations(current, sched_obj, compute_blocks_IDs, cfg)
                 if _join(current, c) not in tried]
        if not cands:
            # Nothing new to add here: jump to another legal schedule (or the original
            # program) that can still be extended; stop only when none is left.
            dead.add(current)
            alive = [b for b in [''] + executed_list
                     if b not in dead and (not b or len(b.split('+')) < cfg["max_schedule_length"])]
            if not alive:
                log.append("Search space exhausted: no legal schedule has untried extensions left.")
                print(log[-1])
                break
            current = rng.choice(alive)
            continue

        candidate = _join(current, _sample_by_kind(rng, cands))
        tried.add(candidate)
        attempts += 1

        # Structurally invalid candidates (wrong level after a prior tiling, unroll
        # past the trip count, interchange of non-perfectly-nested loops, ...) are
        # discarded without counting as an explored schedule.
        try:
            build(candidate)
        except Exception:
            tree_cache.pop(candidate, None)
            logger.debug(f"random search: {candidate} invalid\n{traceback.format_exc()}")
            continue

        response = get_compiler_response(
            candidate, tiramisu_program, annots, compute_blocks_IDs, initial_exec_time,
            explored_scheds, min_runs=min_runs, max_runs=max_runs, time_budget=time_budget,
        )
        outcome = explored_scheds[-1][1]
        log.append(f"[{attempts}] {response}")
        print(f"random search [{executed + (outcome == 'ran')}/{cfg['nb_schedules']} executed, "
              f"attempt {attempts}]: {response}")

        if outcome == 'ran':
            executed += 1
            executed_list.append(candidate)
            current = '' if rng.random() < cfg["restart_prob"] else candidate
        else:
            # Illegal / crashed: kept in explored_scheds for statistics, but never drawn
            # again nor extended; the walk stays on the current schedule.
            tree_cache.pop(candidate, None)

    log.append(f"Random search finished: {executed} executed, {attempts} attempts, "
               f"{len(explored_scheds)} sent to the compiler.")
    print(log[-1])
    return log
