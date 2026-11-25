"""The MOSSImageAccelerate node: a model patch that runs fewer DiT forwards per image."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from . import runtime
from .plan import PRESET_NAMES, PRESETS, WARMUP_RANGE, CacheSpace, Settings, SkipMode

DEFAULT_CUSTOM = PRESETS["balanced"]
DEFAULT_PATTERN = "".join(a.value for a in DEFAULT_CUSTOM.pattern)


class MOSSImageAccelerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MOSSImageAccelerate",
            display_name="MOSS-Image Accelerate",
            category="advanced/model",
            description=(
                "Runs fewer DiT forwards per image on flow-matching models: after a full-CFG warmup, scheduled steps "
                "evaluate the positive prompt only and reuse the last guidance, or skip the model and hold the last "
                "prediction. Inactive below 12 steps, where a partial-denoise run counts at the density of the full "
                "schedule it was cut from and needs at least 6 steps, on models that are not flow matching, and at CFG 1. "
                "The refine preset is for the second, low-denoise pass of an upscale or hires fix; it leaves guidance out of "
                "most steps and changes the content of a full-denoise first pass, so do not use it there."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("preset", options=list(PRESET_NAMES), default="balanced",
                               tooltip="off: unchanged. light, balanced, fast: about 16, 22 and 30 percent fewer forwards at 25 steps. refine: for the second, partial-denoise pass of an upscale or hires fix only (about 33 percent fewer forwards at 9 steps, denoise 0.35); it changes the content of a full-denoise first pass, so do not use it there. custom: the four inputs below."),
                io.Float.Input("warmup", optional=True, default=DEFAULT_CUSTOM.warmup, min=WARMUP_RANGE[0], max=WARMUP_RANGE[1], step=0.01,
                               tooltip="custom only. Share of the steps, from the start, that always run full CFG."),
                io.String.Input("pattern", optional=True, default=DEFAULT_PATTERN,
                                tooltip="custom only. Repeated after the warmup: F full CFG, C positive prompt only with reused guidance, N positive prompt only with no guidance, S no model call. No two S in a row; N and S not together."),
                io.Combo.Input("skip_mode", optional=True, options=SkipMode, default=DEFAULT_CUSTOM.skip_mode,
                               tooltip="custom only. hold: an S step repeats the last prediction. linear: extrapolates the last two over sigma."),
                io.Combo.Input("cache_space", optional=True, options=CacheSpace, default=DEFAULT_CUSTOM.cache_space,
                               tooltip="custom only. x0: hold and reuse guidance on the denoised prediction. v: on the flow velocity."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, preset: str, warmup: float = DEFAULT_CUSTOM.warmup, pattern: str = DEFAULT_PATTERN,
                skip_mode: str = DEFAULT_CUSTOM.skip_mode.value, cache_space: str = DEFAULT_CUSTOM.cache_space.value) -> io.NodeOutput:
        if preset == "custom":
            settings = Settings.parse(warmup, pattern, skip_mode, cache_space)
        elif preset in PRESETS:
            settings = PRESETS[preset]
        else:
            raise ValueError(f"preset must be one of {list(PRESET_NAMES)}, got {preset!r}")
        return io.NodeOutput(runtime.apply(model, preset, settings))


class MOSSImageAccelerateExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MOSSImageAccelerate]


async def comfy_entrypoint() -> MOSSImageAccelerateExtension:
    return MOSSImageAccelerateExtension()
