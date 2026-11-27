"""Which sampling steps run full CFG, cond only, or no DiT at all. Pure Python, no torch."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Callable, Sequence


class Action(enum.Enum):
    """What one sampling step evaluates.

    FULL runs both branches; COND runs the positive prompt and rebuilds the negative one from the last full step's
    guidance; SKIP runs no model and extrapolates from the last evaluated steps; UNGUIDED runs the positive prompt
    only and takes the negative one equal to it, so the step carries no guidance at all.
    """

    FULL = "F"
    COND = "C"
    SKIP = "S"
    UNGUIDED = "N"


class SkipMode(enum.Enum):
    HOLD = "hold"
    LINEAR = "linear"


class CacheSpace(enum.Enum):
    """What a skipped step holds and what the guidance cache stores: the prediction itself, or the flow velocity."""

    X0 = "x0"
    V = "v"


# Below this many steps over the whole schedule the steps are too coarse for reuse, and the 8-step distilled
# recipe runs at CFG 1. A partial run counts at its density: its steps divided by the share of the schedule it covers.
MIN_STEPS = 12
# Measured on ComfyUI's evenly cut schedulers: tails of a grid of 12 or more steps come out at 11.94 equivalent steps and up
# (each scheduler places its cut points and its last step a little differently), tails of shorter grids at 11.63 and below,
# except the six-step tails of an 11-step normal and a 10-step ddim_uniform grid, which start where a 12-step grid's six-step
# tail does. The node sees only sigmas, so the line sits in the gap between 11.63 and 11.94.
EQUIVALENT_TOLERANCE = 0.25
# Fewer steps than this leave no room for a head, a pattern and a full last step, whatever the density.
MIN_PLAN_STEPS = 6
COVERAGE_ITERATIONS = 50
# A sampler that evaluates twice per step repeats a sigma by its third call; full steps until then.
MIN_HEAD_STEPS = 2
MAX_PATTERN = 16
PATTERN_LETTERS = "FCSNfcsn"
WARMUP_RANGE = (0.2, 0.9)
# er_sde moves the first sigma off 1.0 by about 2e-5; a gap is about 1e-2 at 25 steps.
MATCH_TOLERANCE = 0.05

FORWARDS = {Action.FULL: 2, Action.COND: 1, Action.SKIP: 0, Action.UNGUIDED: 1}


@dataclass(frozen=True)
class Settings:
    warmup: float
    pattern: tuple[Action, ...]
    skip_mode: SkipMode = SkipMode.HOLD
    cache_space: CacheSpace = CacheSpace.X0
    stages: tuple[tuple[float, tuple[Action, ...]], ...] = ()
    tail: tuple[Action, ...] = ()
    min_steps: int = MIN_STEPS
    fallback: Settings | None = None

    @classmethod
    def parse(cls, warmup: float, pattern: str, skip_mode: str = SkipMode.HOLD.value,
              cache_space: str = CacheSpace.X0.value) -> Settings:
        lo, hi = WARMUP_RANGE
        if not (isinstance(warmup, (int, float)) and math.isfinite(warmup) and lo <= warmup <= hi):
            raise ValueError(f"warmup must be between {lo} and {hi}, got {warmup!r}")
        text = pattern.strip()
        # Checked before upper(), which turns some other characters into these letters ("ﬀ" into "FF").
        if not set(text) <= set(PATTERN_LETTERS):
            raise ValueError(f"pattern may only hold F, C, S and N, got {pattern!r}")
        text = text.upper()
        if not 1 <= len(text) <= MAX_PATTERN:
            raise ValueError(f"pattern must have 1 to {MAX_PATTERN} letters, got {pattern!r}")
        actions = tuple(Action(c) for c in text)
        # An S step extrapolates the negative branch from the evaluated steps; an N step has none to offer.
        if Action.SKIP in actions and Action.UNGUIDED in actions:
            raise ValueError(f"pattern may not mix S and N: an N step has no negative branch for an S step to extrapolate from, got {pattern!r}")
        if "SS" in text + text[:1]:
            raise ValueError(f"pattern repeats, so it may not hold two S in a row, also across its end: {pattern!r}")
        try:
            mode = SkipMode(skip_mode)
        except ValueError:
            raise ValueError(f"skip_mode must be one of {[m.value for m in SkipMode]}, got {skip_mode!r}") from None
        try:
            space = CacheSpace(cache_space)
        except ValueError:
            raise ValueError(f"cache_space must be one of {[s.value for s in CacheSpace]}, got {cache_space!r}") from None
        return cls(float(warmup), actions, mode, space)


_FAST_PRESET = Settings.parse(0.36, "CSF", SkipMode.LINEAR.value)

PRESETS: dict[str, Settings | None] = {
    "off": None,
    "light": Settings.parse(0.32, "CF"),
    "balanced": Settings.parse(0.32, "CCF"),
    "fast": _FAST_PRESET,
    # Calibrated on 50-step deterministic flow schedules (MOSS-Image-Base: euler/simple, CFG 3.0, shift 5, SLG 22).
    # Keeps a 12-step full-CFG warmup to lock composition and hand geometry, uses two-step linear x0 skips only in
    # the dense high-noise plateau (steps 12..35), tightens to single skips in the mid-detail band (36..43), and
    # finishes with an unguided positive-only polish tail ending on full CFG (CNNNNF: 48/100 forwards at 50 steps).
    # Falls back to `fast` below 30 steps so coarse or partial-denoise runs never skip two steps in a row.
    "fast_50": Settings(
        warmup=0.24,
        pattern=(Action.SKIP, Action.COND, Action.SKIP, Action.FULL),
        skip_mode=SkipMode.LINEAR,
        cache_space=CacheSpace.X0,
        stages=(
            (0.72, (Action.SKIP, Action.SKIP, Action.COND, Action.SKIP, Action.SKIP, Action.FULL)),
            (0.88, (Action.SKIP, Action.COND, Action.SKIP, Action.COND, Action.SKIP, Action.COND, Action.SKIP, Action.FULL)),
        ),
        tail=(Action.COND, Action.UNGUIDED, Action.UNGUIDED, Action.UNGUIDED, Action.UNGUIDED, Action.FULL),
        min_steps=30,
        fallback=_FAST_PRESET,
    ),
    # For a second, partial-denoise pass (upscale, hires fix). The base image is already composed, so after a short
    # warmup every step runs the positive prompt only. On a full-denoise pass it changes the content.
    "refine": Settings.parse(0.2, "N"),
}
PRESET_NAMES = (*PRESETS, "custom")


def head_steps(steps: int, warmup: float) -> int:
    return max(MIN_HEAD_STEPS, math.ceil(warmup * steps - 1e-9))


def coverage_of(first_sigma: float, sigma_at_percent: Callable[[float], float]) -> float:
    """Share of the model's whole schedule that a run starting at first_sigma covers, in (0, 1].

    sigma_at_percent is the model's percent_to_sigma: 0 is the start of a full run, 1 its end, falling in
    between. Anything that cannot be read as such a curve gives 1.0, which plans the run on its own step count.
    """
    try:
        target = float(first_sigma)
        top, bottom = float(sigma_at_percent(0.0)), float(sigma_at_percent(1.0))
        # Outside the curve's range (the start of a full run, or no sigma it reaches) the run is not located.
        if not (math.isfinite(target) and math.isfinite(top) and math.isfinite(bottom) and bottom < target < top):
            return 1.0
        lo, hi, s_lo, s_hi = 0.0, 1.0, top, bottom
        for _ in range(COVERAGE_ITERATIONS):
            mid = (lo + hi) / 2
            s_mid = float(sigma_at_percent(mid))
            if not (math.isfinite(s_mid) and s_hi <= s_mid <= s_lo):
                return 1.0
            if s_mid > target:
                lo, s_lo = mid, s_mid
            else:
                hi, s_hi = mid, s_mid
    except Exception:
        return 1.0
    return min(1.0, 1.0 - (lo + hi) / 2)


def equivalent_steps(steps: int, coverage: float) -> float:
    """The step count of the whole schedule a run with this coverage was cut from, as measured.

    KSampler cuts a partial denoise from a grid of int(steps / denoise) steps, but coverage is read off a
    continuous curve, so the quotient lands a little off a whole number. It is not rounded: plan_steps gates on
    it with a margin, and the log prints this same number.
    """
    return steps / coverage


def plan_steps(steps: int, settings: Settings | None, coverage: float = 1.0) -> tuple[Action, ...]:
    """The head and the last step run full CFG; the pattern repeats in between.

    coverage is the share of the whole schedule this run covers (coverage_of). A partial-denoise tail
    is as fine-grained as the full run it was cut from, so it is planned when its equivalent steps reach
    MIN_STEPS, not steps alone.

    The last step's prediction is the image for samplers that end on sigma 0, so it is never
    approximated, and the step before it is never skipped so that the last input comes from a fresh
    evaluation.
    """
    if not (math.isfinite(coverage) and 0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    if settings is None or steps < MIN_PLAN_STEPS or equivalent_steps(steps, coverage) < MIN_STEPS - EQUIVALENT_TOLERANCE:
        return (Action.FULL,) * steps
    if settings.stages:
        head = head_steps(steps, settings.warmup)
        tail_start = steps - len(settings.tail)
        # Phased high-step schedules (e.g. fast_50) assume a full dense schedule, not a short or partial-denoise tail.
        if steps < settings.min_steps or coverage < 0.95 or tail_start <= head:
            return plan_steps(steps, settings.fallback, coverage)
        plan: list[Action] = [Action.FULL] * head
        stage_start = head
        for idx, (end_frac, pat) in enumerate(settings.stages):
            stage_end = tail_start if idx == len(settings.stages) - 1 else min(tail_start, max(stage_start, round(end_frac * steps)))
            for i in range(stage_start, stage_end):
                action = pat[(i - stage_start) % len(pat)]
                if action is Action.SKIP and len(plan) >= 2 and plan[-1] is Action.SKIP and plan[-2] is Action.SKIP:
                    action = Action.COND
                plan.append(action)
            stage_start = stage_end
        plan.extend(settings.tail)
        if not settings.tail:
            plan[-1] = Action.FULL
        return tuple(plan)
    head = head_steps(steps, settings.warmup)
    plan = []
    for i in range(steps):
        if i < head or i == steps - 1:
            plan.append(Action.FULL)
            continue
        action = settings.pattern[(i - head) % len(settings.pattern)]
        if action is Action.SKIP and (i == steps - 2 or plan[-1] is Action.SKIP):
            action = Action.COND
        plan.append(action)
    return tuple(plan)


def forward_count(plan: Sequence[Action], guided: bool = True) -> int:
    if guided:
        return sum(FORWARDS[a] for a in plan)
    return sum(a is not Action.SKIP for a in plan)


def saved_fraction(plan: Sequence[Action], guided: bool = True) -> float:
    full = len(plan) * (2 if guided else 1)
    return 0.0 if full == 0 else 1.0 - forward_count(plan, guided) / full


def parse_schedule(values: Sequence[float]) -> tuple[float, ...] | None:
    """The sigmas of one sampling run, or None when they cannot carry a plan."""
    schedule = tuple(float(v) for v in values)
    if len(schedule) < 2 or not all(math.isfinite(v) for v in schedule):
        return None
    if any(b >= a for a, b in zip(schedule, schedule[1:])):
        return None
    return schedule


def locate_step(sigma: float, schedule: Sequence[float]) -> int | None:
    """Index of the step whose sigma this is, within a fraction of its gaps to the neighbours."""
    steps = len(schedule) - 1
    if not math.isfinite(sigma) or steps < 1:
        return None
    i = min(range(steps), key=lambda k: abs(schedule[k] - sigma))
    gaps = [schedule[i] - schedule[i + 1]]
    if i > 0:
        gaps.append(schedule[i - 1] - schedule[i])
    return i if abs(schedule[i] - sigma) <= MATCH_TOLERANCE * min(gaps) else None


class Cursor:
    """Follows one run's model calls along the schedule; any call off the plan's path makes the rest full."""

    def __init__(self, schedule: Sequence[float], plan: Sequence[Action]):
        if len(plan) != len(schedule) - 1:
            raise ValueError("plan and schedule disagree on the number of steps")
        self.schedule = tuple(schedule)
        self.plan = tuple(plan)
        self.last = -1
        self.irregular_at: int | None = None
        self.calls = 0

    @property
    def irregular(self) -> bool:
        return self.irregular_at is not None

    def advance(self, sigma: float) -> Action:
        self.calls += 1
        if self.irregular:
            return Action.FULL
        i = locate_step(sigma, self.schedule)
        if i is None or i != self.last + 1:
            self.irregular_at = self.calls
            return Action.FULL
        self.last = i
        return self.plan[i]
