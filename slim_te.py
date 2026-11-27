"""Zero-mmap streaming Slim Text Encoder loaders for MOSS-Image (Qwen3.5-9B).

Provides two production loaders designed for consumer 8-16 GB GPUs:
- Row-W8A16 Slim (~7.94 GB, per-row symmetric INT8 weights + BF16 activations, dead lm_head/visual/mtp stripped)
  using buffered seek+readinto I/O so multi-GB safetensors files never reserve Windows MapViewOfFile pagefile commit.
- Slim W4A8/FP8 (~4.93 GB) with header-only safetensors key inspection and automatic dead-tower pruning.
"""

from __future__ import annotations

import json
import pathlib
import struct
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}


def read_safetensors_header(path: str | pathlib.Path) -> tuple[dict[str, Any], int]:
    """Read only the JSON header of a safetensors file without mapping or loading tensor payloads."""
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise EOFError(f"Truncated safetensors header length in {path}")
        (hdr_len,) = struct.unpack("<Q", raw_len)
        raw_hdr = fh.read(hdr_len)
        if len(raw_hdr) != hdr_len:
            raise EOFError(f"Truncated safetensors JSON header in {path}")
        return json.loads(raw_hdr.decode("utf-8")), 8 + hdr_len


class SafetensorsStreamReader:
    """Zero-mmap safetensors reader using plain file seek + readinto into pre-allocated CPU tensors."""

    def __init__(self, path: str | pathlib.Path):
        self._path = pathlib.Path(path)
        self._f = open(self._path, "rb", buffering=8 * 1024 * 1024)
        raw_len = self._f.read(8)
        if len(raw_len) != 8:
            self._f.close()
            raise EOFError(f"Truncated safetensors header length in {self._path}")
        (hdr_len,) = struct.unpack("<Q", raw_len)
        raw_hdr = self._f.read(hdr_len)
        if len(raw_hdr) != hdr_len:
            self._f.close()
            raise EOFError(f"Truncated safetensors JSON header in {self._path}")
        self.header: dict[str, Any] = json.loads(raw_hdr.decode("utf-8"))
        self.data_base = 8 + hdr_len

    def __enter__(self) -> SafetensorsStreamReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def keys(self) -> list[str]:
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key: str, device: str | torch.device = "cpu") -> torch.Tensor:
        entry = self.header[key]
        s_off, e_off = entry["data_offsets"]
        dt = _DTYPE_MAP[entry["dtype"]]
        t = torch.empty(list(entry["shape"]), dtype=dt, device="cpu")
        total = e_off - s_off
        if total > 0:
            self._f.seek(self.data_base + s_off)
            mv = memoryview(t.view(torch.uint8).reshape(-1).numpy())
            pos = 0
            try:
                while pos < total:
                    n = self._f.readinto(mv[pos:])
                    if not n:
                        raise EOFError(f"Unexpected EOF reading {key} at {pos}/{total}")
                    pos += n
            finally:
                mv.release()
        if str(device) != "cpu":
            t = t.to(device=device)
        return t


class RowW8A16Linear(nn.Module):
    """Per-row symmetric INT8 weight + BF16 [out, 1] scale with native BF16 activations.

    Uses nn.Parameter(requires_grad=False) without comfy_cast_weights so ComfyUI's ModelPatcher
    tracks exact 1-byte INT8 VRAM size and swaps modules cleanly between GPU and host RAM.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.out_features, self.in_features = qweight.shape
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        w = self.qweight.to(dtype=dt) * self.scale.to(dtype=dt)
        b = self.bias.to(dtype=dt) if self.bias is not None else None
        return F.linear(x, w, b)


class RowW8A16Embedding(nn.Module):
    """Per-row symmetric INT8 token embedding + BF16 [vocab, 1] scale."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, idx: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        qw = F.embedding(idx, self.qweight)
        sc = F.embedding(idx, self.scale)
        return qw.to(dtype=out_dtype) * sc.to(dtype=out_dtype)


def _assign_leaf_tensor(root_mod: nn.Module, rel_key: str, tensor: torch.Tensor) -> None:
    """Assign a single parameter/buffer onto its exact leaf submodule without invoking load_state_dict.

    Calling layer.load_state_dict(sub_sd, strict=False, assign=True) on a partially-meta transformer
    block causes PyTorch/ComfyUI hooks to materialize sibling meta Linear weights (~22 GB on 9B TE).
    Assigning leaves directly after swapping quantized modules keeps private CPU commit near zero.
    """
    parts = rel_key.split(".")
    parent: nn.Module = root_mod
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    if attr in parent._buffers:
        parent._buffers[attr] = tensor
    else:
        setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))


def load_row_w8a16_clip(clip_name: str):
    """Load a stripped per-row INT8 + BF16-activation Qwen3.5-9B text encoder for MOSS-Image."""
    import contextlib
    import comfy.hooks
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sd
    import comfy.text_encoders.moss_image as moss_te
    import comfy.utils
    import folder_paths

    ckpt_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    load_device = comfy.model_management.text_encoder_device()
    offload_device = comfy.model_management.text_encoder_offload_device()

    aimdo_enabled = bool(
        getattr(getattr(comfy, "memory_management", None), "aimdo_enabled", False)
    )
    fast_disk = False

    with contextlib.ExitStack() as stack:
        if aimdo_enabled:
            # ComfyUI v0.46+ aimdo path: file-backed ModelMMAP + TensorFileSlice (~0 GB CPU commit).
            sd = comfy.utils.load_torch_file(ckpt_path)
            all_keys = set(sd.keys())
            try:
                import comfy.storage

                fast_disk = bool(comfy.storage.state_dict_fast_disk(sd))
            except Exception:
                fast_disk = False

            def _get_tensor(key: str) -> torch.Tensor:
                t = sd[key]
                if str(offload_device) != "cpu":
                    t = t.to(device=offload_device)
                return t
        else:
            # Legacy / non-DynamicVRAM path: zero-mmap streaming readinto avoids Windows pagefile spikes.
            sf = stack.enter_context(SafetensorsStreamReader(ckpt_path))
            all_keys = set(sf.keys())

            def _get_tensor(key: str) -> torch.Tensor:
                return sf.get_tensor(key, device=offload_device)

        if "model.language_model.embed_tokens.weight_scale" not in all_keys:
            raise ValueError(
                f"{clip_name} is missing row_w8a16 .weight_scale tensors; "
                "use MOSSSlimCLIPLoader for W4A8/FP8 checkpoints."
            )

        te_cls = moss_te.te(dtype_llama=torch.bfloat16, mtp=False)
        te_model = te_cls(
            device="meta",
            dtype=torch.bfloat16,
            model_options={"qwen35_9b_model_config": {"lm_head": False, "mtp": False}},
        )
        tr = te_model.qwen35_9b.transformer
        tr.visual = None
        if hasattr(tr, "model") and hasattr(tr.model, "lm_head"):
            delattr(tr.model, "lm_head")

        eq = _get_tensor("model.language_model.embed_tokens.weight")
        esc = _get_tensor("model.language_model.embed_tokens.weight_scale")
        tr.model.embed_tokens = RowW8A16Embedding(eq, esc)

        nw = _get_tensor("model.language_model.norm.weight")
        _assign_leaf_tensor(tr.model.norm, "weight", nw)

        layer_buckets: list[list[str]] = [[] for _ in range(len(tr.model.layers))]
        prefix_root = "model.language_model.layers."
        for k in all_keys:
            if k.startswith(prefix_root):
                rest = k[len(prefix_root):]
                dot = rest.find(".")
                if dot > 0:
                    idx_s = rest[:dot]
                    if idx_s.isdigit():
                        li = int(idx_s)
                        if 0 <= li < len(layer_buckets):
                            layer_buckets[li].append(rest[dot + 1:])

        for i, layer in enumerate(tr.model.layers):
            pfx = f"model.language_model.layers.{i}."
            rel_set = set(layer_buckets[i])
            quant_mpaths = {
                rk[:-len(".weight_scale")] for rk in rel_set if rk.endswith(".weight_scale")
            }

            # Swap quantized Linear modules FIRST before touching non-quant leaves.
            for mpath in sorted(quant_mpaths):
                qw = _get_tensor(f"{pfx}{mpath}.weight")
                sc = _get_tensor(f"{pfx}{mpath}.weight_scale")
                bk = f"{mpath}.bias"
                if bk in rel_set:
                    b_raw = _get_tensor(f"{pfx}{bk}")
                    b_ten = b_raw if b_raw.dtype == torch.bfloat16 else b_raw.to(dtype=torch.bfloat16)
                else:
                    b_ten = None
                new_mod = RowW8A16Linear(qw, sc, bias=b_ten)
                parts = mpath.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_mod)

            # Assign remaining tiny norm/conv/scalar leaves directly without layer.load_state_dict().
            for rk in sorted(rel_set):
                if rk.endswith(".weight_scale"):
                    continue
                if rk.endswith(".weight") and rk[:-len(".weight")] in quant_mpaths:
                    continue
                if rk.endswith(".bias") and rk[:-len(".bias")] in quant_mpaths:
                    continue
                _assign_leaf_tensor(layer, rk, _get_tensor(pfx + rk))

    meta_left = [n for n, p in te_model.named_parameters() if getattr(p, "is_meta", False)] + [
        n for n, b in te_model.named_buffers() if getattr(b, "is_meta", False)
    ]
    if meta_left:
        raise RuntimeError(f"MOSSRowW8A16CLIPLoader left meta tensors: {meta_left[:8]}")

    te_model.device = offload_device
    te_model.dtypes = {torch.bfloat16}
    comfy.model_management.archive_model_dtypes(te_model)

    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = te_model
    clip.tokenizer = moss_te.MOSSImageTokenizer(
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        tokenizer_data={},
    )
    clip.tokenizer_metadata = {}
    patcher_cls = getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )
    try:
        clip.patcher = patcher_cls(
            te_model,
            load_device=load_device,
            offload_device=offload_device,
            fast_disk=fast_disk,
        )
    except TypeError:
        clip.patcher = patcher_cls(
            te_model,
            load_device=load_device,
            offload_device=offload_device,
        )
    clip.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    clip.patcher.is_clip = True
    clip.apply_hooks_to_conds = None
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.tokenizer_options = {}
    return clip


def load_slim_clip(clip_name: str, drop_vision_tower: bool = True):
    """Load a slim W4A8/FP8 Qwen3.5-9B text encoder for MOSS-Image without loading dead towers."""
    import comfy.sd
    import folder_paths

    ckpt_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    header, _ = read_safetensors_header(ckpt_path)
    keys = [k for k in header.keys() if k != "__metadata__"]
    has_lm_head = ("lm_head.weight" in header) or ("model.lm_head.weight" in header)
    has_visual = any(k.startswith(("model.visual.", "visual.")) for k in keys)

    model_options = {
        "qwen35_9b_model_config": {
            "lm_head": bool(has_lm_head),
            "mtp": False,
        }
    }
    clip = comfy.sd.load_clip(
        ckpt_paths=[ckpt_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
        model_options=model_options,
    )

    if drop_vision_tower or not has_visual:
        try:
            te_wrapper = getattr(clip.cond_stage_model, "qwen35_9b", None) or getattr(
                clip.cond_stage_model, "clip", None
            )
            if te_wrapper is not None and hasattr(te_wrapper, "transformer"):
                tr = te_wrapper.transformer
                if hasattr(tr, "visual") and tr.visual is not None:
                    tr.visual = None
                if not has_lm_head and hasattr(tr, "model") and hasattr(tr.model, "lm_head"):
                    delattr(tr.model, "lm_head")
        except Exception:
            pass

    return clip
