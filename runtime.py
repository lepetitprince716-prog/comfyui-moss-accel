"""The sampling-time half of MOSS-Image Accelerate: one Run per sampling, driven from the calc-cond-batch hook."""

from __future__ import annotations

import logging
import math
from collections import Counter, deque

import torch

import comfy.model_patcher
import comfy.model_sampling
import comfy.patcher_extension
import comfy.samplers

from .plan import (Action, CacheSpace, Cursor, Settings, SkipMode, coverage_of, equivalent_steps, forward_count,
                   parse_schedule, plan_steps)

SETTINGS_KEY = "moss_image_accelerate"
RUN_KEY = "moss_image_accelerate_run"
log = logging.getLogger("moss_image_accelerate")


def _calc_cond_batch(args):
    return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])


class Evaluation:
    """One evaluated step in the run's cache space; uncond is None when the sampler ran cond only (CFG 1)."""

    def __init__(self, sigma: float, cond: torch.Tensor, uncond: torch.Tensor | None):
        self.sigma = sigma
        self.cond = cond
        self.uncond = uncond


class Run:
    def __init__(self, settings: Settings, preset: str, sigma_at_percent=None):
        self.settings = settings
        self.preset = preset
        # The model's percent_to_sigma, to tell a partial-denoise tail from a coarse run; None counts as a full run.
        self.sigma_at_percent = sigma_at_percent
        self.coverage = 1.0
        self.disabled: str | None = None
        self.busy = False
        self.calls = 0
        self.schedule: tuple[float, ...] | None = None
        self.schedule_source = None
        self.cursor: Cursor | None = None
        self.history: deque[Evaluation] = deque(maxlen=2)
        self.guidance: torch.Tensor | None = None
        self.layout = None
        self.served: Counter[Action] = Counter()
        self.forwards = 0
        self.guided_calls = 0
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)
            log.info("MOSS-Image Accelerate: %s", text)

    def _follow(self, sample_sigmas) -> Cursor | None:
        if sample_sigmas is None:
            self.note("no sample_sigmas in transformer_options, inactive for this sampling")
            return None
        if sample_sigmas is not self.schedule_source:
            self.schedule_source = sample_sigmas
            values = sample_sigmas.tolist() if isinstance(sample_sigmas, torch.Tensor) else list(sample_sigmas)
            schedule = parse_schedule(values)
            if schedule is None:
                self.note("sample_sigmas are not a finite, strictly falling schedule, inactive for this sampling")
                self.schedule = self.cursor = None
                return None
            if schedule != self.schedule:
                steps = len(schedule) - 1
                self.coverage = 1.0 if self.sigma_at_percent is None else coverage_of(schedule[0], self.sigma_at_percent)
                plan = plan_steps(steps, self.settings, self.coverage)
                if all(a is Action.FULL for a in plan):
                    self.note(f"{steps} steps ({self._density(steps)}) is below the minimum, inactive for this sampling")
                self.schedule, self.cursor = schedule, Cursor(schedule, plan)
                self._forget()
        return self.cursor

    def _density(self, steps: int) -> str:
        return f"coverage {self.coverage:.2f}, {equivalent_steps(steps, self.coverage):.2f} equivalent steps"

    def _forget(self) -> None:
        self.history.clear()
        self.guidance = None
        self.layout = None

    def _action(self, args) -> Action:
        if self.disabled is not None:
            return Action.FULL
        cursor = self._follow(args["model_options"].get("transformer_options", {}).get("sample_sigmas"))
        if cursor is None:
            return Action.FULL
        sigmas = args["sigma"].flatten()
        sigma = float(sigmas[0])
        if not bool((sigmas == sigmas[0]).all()) or sigma <= 0.0:
            sigma = float("nan")
        was_irregular = cursor.irregular
        action = cursor.advance(sigma)
        if cursor.irregular and not was_irregular:
            self.note(f"call {cursor.irregular_at} is off the schedule (a sampler that evaluates more than once per step), full CFG from here on")
        return action

    def _to_space(self, denoised: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.settings.cache_space is CacheSpace.X0:
            return denoised.clone()
        return (x - denoised) / sigma

    def _from_space(self, value: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.settings.cache_space is CacheSpace.X0:
            return value.clone()
        return x - sigma * value

    def evaluate(self, args, compute):
        self.calls += 1
        x = args["input"]
        conds = args["conds"]
        guided = conds[1] is not None
        action = self._action(args)
        layout = (x.shape, x.dtype, x.device)
        if layout != self.layout:
            self._forget()
            self.layout = layout
        sigma_value = float(args["sigma"].flatten()[0])
        sigma = args["sigma"].reshape((-1,) + (1,) * (x.ndim - 1)).to(x.dtype)

        if action is Action.SKIP:
            predicted = self._predict(sigma_value)
            if predicted is not None and (predicted[1] is not None or not guided):
                cond_q, uncond_q = predicted
                self.served[Action.SKIP] += 1
                self.guided_calls += guided
                uncond_out = self._from_space(uncond_q, x, sigma) if guided else torch.zeros_like(x)
                return [self._from_space(cond_q, x, sigma), uncond_out]
            action = Action.COND

        if action is Action.UNGUIDED and guided:
            # Neither the held guidance nor the history is touched: the negative branch here is not a measurement.
            out = compute({**args, "conds": [conds[0], None]})
            self.served[Action.UNGUIDED] += 1
            self.forwards += 1
            self.guided_calls += 1
            return [out[0], out[0].clone()]

        if action is Action.COND and guided and self.guidance is not None:
            out = compute({**args, "conds": [conds[0], None]})
            cond_q = self._to_space(out[0], x, sigma)
            uncond_q = cond_q - self.guidance
            self.served[Action.COND] += 1
            self.forwards += 1
            self.guided_calls += 1
            self.history.append(Evaluation(sigma_value, cond_q, uncond_q))
            return [out[0], self._from_space(uncond_q, x, sigma)]

        out = compute(args)
        self.served[Action.FULL] += 1
        self.forwards += 2 if guided else 1
        self.guided_calls += guided
        if sigma_value > 0.0:
            cond_q = self._to_space(out[0], x, sigma)
            uncond_q = self._to_space(out[1], x, sigma) if guided else None
            self.guidance = cond_q - uncond_q if guided else None
            self.history.append(Evaluation(sigma_value, cond_q, uncond_q))
        return out

    def _predict(self, sigma: float):
        if not self.history:
            return None
        last = self.history[-1]
        if self.settings.skip_mode is SkipMode.HOLD or len(self.history) < 2:
            return last.cond, last.uncond
        first = self.history[0]
        if first.sigma == last.sigma:
            return last.cond, last.uncond
        k = (sigma - last.sigma) / (last.sigma - first.sigma)
        cond = last.cond + k * (last.cond - first.cond)
        if last.uncond is None or first.uncond is None:
            return cond, last.uncond
        return cond, last.uncond + k * (last.uncond - first.uncond)

    def summary(self) -> str:
        if self.disabled is not None:
            return f"inactive: {self.disabled}"
        if self.calls == 0:
            return "never called: a patch applied after this node replaced sampler_calc_cond_batch_function"
        if self.cursor is None:
            return "inactive"
        steps = len(self.cursor.plan)
        guided = self.guided_calls > 0
        baseline = self.calls * (2 if guided else 1)
        planned = "all full, sampler is off the schedule" if self.cursor.irregular else f"plan {forward_count(self.cursor.plan, guided)}"
        plan_text = "".join(a.value for a in self.cursor.plan)
        unguided = f" N{self.served[Action.UNGUIDED]}" if Action.UNGUIDED in self.cursor.plan else ""
        return (f"preset {self.preset}, {steps} steps ({self._density(steps)}), plan {plan_text},"
                f" forwards {self.forwards} of {baseline}"
                f" ({planned}), served F{self.served[Action.FULL]} C{self.served[Action.COND]} S{self.served[Action.SKIP]}{unguided}")


def make_calc_function(previous):
    compute = previous if previous is not None else _calc_cond_batch

    def calc_cond_batch(args):
        run = args["model_options"].get("transformer_options", {}).get(RUN_KEY)
        # A patch that chains onto this function and is chained onto by it would bring the call back here.
        if run is None or run.busy:
            return compute(args)
        run.busy = True
        try:
            return run.evaluate(args, compute)
        finally:
            run.busy = False

    calc_cond_batch.moss_accel_previous = previous
    return calc_cond_batch


def _no_guidance(guider) -> str | None:
    """Why this guider does no classifier-free guidance, or None when it does.

    At CFG 1 ComfyUI leaves the negative prompt out, so there is no second forward to save, and a distilled
    few-step recipe has no steps to spare for skipping. A guider with no cfg (BasicGuider keeps the default 1.0
    of its base class, a third-party guider may have none) does no guidance either.
    """
    cfg = getattr(guider, "cfg", None)
    try:
        value = float(cfg)
    except (TypeError, ValueError):
        return f"{type(guider).__name__} has no numeric cfg, so it does no classifier-free guidance"
    if math.isclose(value, 1.0):
        return f"guider {type(guider).__name__} has cfg {value:g}: no classifier-free guidance to reuse, every step runs in full"
    return None


def outer_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original = guider.model_options
    entry = original.get("transformer_options", {}).get(SETTINGS_KEY)
    if entry is None:
        return executor(*args, **kwargs)
    options = comfy.model_patcher.create_model_options_clone(original)
    preset, settings = entry
    model_sampling = guider.model_patcher.get_model_object("model_sampling")
    run = Run(settings, preset, getattr(model_sampling, "percent_to_sigma", None))
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        run.disabled = f"model_sampling {type(model_sampling).__name__} is not flow matching (CONST)"
    elif (reason := _no_guidance(guider)) is not None:
        run.disabled = reason
        run.note(f"{reason}, inactive for this sampling")
    elif type(guider.model_patcher.model).__name__ != "MOSSImage":
        run.note(f"{type(guider.model_patcher.model).__name__} is not MOSS-Image; presets were set for MOSS-Image")
    options["transformer_options"][RUN_KEY] = run
    guider.model_options = options
    try:
        return executor(*args, **kwargs)
    finally:
        guider.model_options = original
        log.info("MOSS-Image Accelerate: %s", run.summary())


def apply(model, preset: str, settings: Settings | None):
    """A clone of the ModelPatcher with the accelerator attached; the input patcher is left as it was."""
    patched = model.clone()
    current = patched.model_options.get("sampler_calc_cond_batch_function")
    previous = current
    while previous is not None and hasattr(previous, "moss_accel_previous"):
        previous = previous.moss_accel_previous
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, SETTINGS_KEY)
    if settings is None:
        if current is not previous:
            # "off" behind an earlier accelerator: take that one out again.
            if previous is None:
                patched.model_options.pop("sampler_calc_cond_batch_function", None)
            else:
                patched.set_model_sampler_calc_cond_batch_function(previous)
        patched.model_options.get("transformer_options", {}).pop(SETTINGS_KEY, None)
        return patched
    patched.set_model_sampler_calc_cond_batch_function(make_calc_function(previous))
    patched.model_options.setdefault("transformer_options", {})[SETTINGS_KEY] = (preset, settings)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, SETTINGS_KEY, outer_sample_wrapper)
    return patched
