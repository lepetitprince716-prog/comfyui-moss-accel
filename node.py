"""The MOSSImageAccelerate node: a model patch that runs fewer DiT forwards per image."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
import folder_paths

from . import runtime, slim_te
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
                               tooltip="off: unchanged. light, balanced, fast: about 16, 22 and 30 percent fewer forwards at 25 steps. fast_50: phased 50-step Base schedule (48/100 forwards, ~2x speedup, falls back to fast below 30 steps). refine: for the second, partial-denoise pass of an upscale or hires fix only (about 33 percent fewer forwards at 9 steps, denoise 0.35); it changes the content of a full-denoise first pass, so do not use it there. custom: the four inputs below."),
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


class MOSSRowW8A16CLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = folder_paths.get_filename_list("text_encoders")
        default_name = (
            "qwen3.5_9b_row_w8a16_slim.safetensors"
            if "qwen3.5_9b_row_w8a16_slim.safetensors" in files
            else (files[0] if files else "qwen3.5_9b_row_w8a16_slim.safetensors")
        )
        return io.Schema(
            node_id="MOSSRowW8A16CLIPLoader",
            display_name="Load MOSS Row-W8A16 Slim Text Encoder",
            category="advanced/loaders",
            description=(
                "Streams a stripped per-row INT8 + BF16-activation Qwen3.5-9B text encoder (~7.94 GB) "
                "for MOSS-Image without memory-mapping the full file, keeping full GPU encode speed and "
                "automatic VRAM offload on 16 GB cards."
            ),
            inputs=[
                io.Combo.Input("clip_name", options=files or [default_name], default=default_name),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name: str) -> io.NodeOutput:
        return io.NodeOutput(slim_te.load_row_w8a16_clip(clip_name))


class MOSSSlimCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = folder_paths.get_filename_list("text_encoders")
        default_name = (
            "qwen3.5_9b_w4a8_slim.safetensors"
            if "qwen3.5_9b_w4a8_slim.safetensors" in files
            else (files[0] if files else "qwen3.5_9b_w4a8_slim.safetensors")
        )
        return io.Schema(
            node_id="MOSSSlimCLIPLoader",
            display_name="Load MOSS Slim Text Encoder (W4A8 / FP8)",
            category="advanced/loaders",
            description=(
                "Loads a slim W4A8/FP8 Qwen3.5-9B text encoder (~4.93 GB) for MOSS-Image using header-only "
                "key inspection and drops unused vision/lm_head/mtp weights."
            ),
            inputs=[
                io.Combo.Input("clip_name", options=files or [default_name], default=default_name),
                io.Boolean.Input("drop_vision_tower", default=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name: str, drop_vision_tower: bool = True) -> io.NodeOutput:
        return io.NodeOutput(slim_te.load_slim_clip(clip_name, drop_vision_tower=drop_vision_tower))


class MOSSImageAccelerateExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MOSSImageAccelerate, MOSSRowW8A16CLIPLoader, MOSSSlimCLIPLoader]


async def comfy_entrypoint() -> MOSSImageAccelerateExtension:
    return MOSSImageAccelerateExtension()
