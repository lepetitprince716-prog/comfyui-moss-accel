"""Unit tests for the zero-mmap Slim Text Encoder reader and Row-W8A16 modules."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import struct
import sys
import tempfile
import unittest

PROJECT = pathlib.Path(__file__).resolve().parent.parent
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

    _spec = importlib.util.spec_from_file_location("moss_accel_slim_te", PROJECT / "slim_te.py")
    slim_te = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = slim_te
    _spec.loader.exec_module(slim_te)


def _write_minimal_safetensors(path: pathlib.Path, tensors: dict[str, "torch.Tensor"]) -> None:
    dtype_to_name = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.int32: "I32",
        torch.int64: "I64",
    }
    header = {"__metadata__": {"format": "pt"}}
    blobs: list[bytes] = []
    offset = 0
    for name, ten in tensors.items():
        raw = ten.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype_to_name[ten.dtype],
            "shape": list(ten.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for b in blobs:
            fh.write(b)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class SlimTETest(unittest.TestCase):
    def test_stream_reader_round_trip_across_dtypes(self):
        gen = torch.Generator().manual_seed(42)
        expected = {
            "a.weight": torch.randint(-127, 128, (8, 16), dtype=torch.int8, generator=gen),
            "a.weight_scale": torch.randn((8, 1), generator=gen).to(torch.bfloat16),
            "b.norm": torch.randn((16,), dtype=torch.float32, generator=gen),
            "c.idx": torch.arange(6, dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "mini.safetensors"
            _write_minimal_safetensors(p, expected)
            hdr, base = slim_te.read_safetensors_header(p)
            self.assertGreater(base, 8)
            self.assertEqual(set(k for k in hdr if k != "__metadata__"), set(expected.keys()))
            with slim_te.SafetensorsStreamReader(p) as sf:
                self.assertEqual(set(sf.keys()), set(expected.keys()))
                for k, want in expected.items():
                    got = sf.get_tensor(k)
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertEqual(tuple(got.shape), tuple(want.shape))
                    self.assertTrue(torch.equal(got, want))

    def test_stream_reader_rejects_truncated_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "bad.safetensors"
            p.write_bytes(b"\x10\x00\x00\x00")
            with self.assertRaises(EOFError):
                slim_te.read_safetensors_header(p)

    def test_row_w8a16_linear_matches_dense_bf16_and_tracks_int8_params(self):
        gen = torch.Generator().manual_seed(7)
        qw = torch.randint(-127, 128, (12, 20), dtype=torch.int8, generator=gen)
        sc = (torch.rand((12, 1), generator=gen) * 0.05).to(torch.bfloat16)
        bias = torch.randn((12,), generator=gen).to(torch.bfloat16)
        x = torch.randn((2, 5, 20), generator=gen).to(torch.bfloat16)

        mod = slim_te.RowW8A16Linear(qw, sc, bias=bias)
        self.assertIsInstance(mod.qweight, torch.nn.Parameter)
        self.assertFalse(mod.qweight.requires_grad)
        self.assertEqual(mod.qweight.dtype, torch.int8)
        self.assertFalse(hasattr(mod, "comfy_cast_weights"))

        got = mod(x)
        want = F.linear(x, qw.to(torch.bfloat16) * sc, bias)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_row_w8a16_embedding_matches_dense_bf16(self):
        gen = torch.Generator().manual_seed(11)
        qw = torch.randint(-127, 128, (32, 16), dtype=torch.int8, generator=gen)
        sc = (torch.rand((32, 1), generator=gen) * 0.05).to(torch.bfloat16)
        idx = torch.tensor([[0, 5, 17, 31]], dtype=torch.long)

        emb = slim_te.RowW8A16Embedding(qw, sc)
        got = emb(idx, out_dtype=torch.bfloat16)
        want = F.embedding(idx, qw.to(torch.bfloat16) * sc)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_assign_leaf_tensor_leaves_sibling_meta_modules_untouched(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(8, device="meta")
                self.proj = torch.nn.Linear(8, 8, bias=False, device="meta")
                self.register_buffer("scale_buf", torch.empty(4, device="meta"))

        blk = Block()
        w = torch.ones(8, dtype=torch.bfloat16)
        buf = torch.arange(4, dtype=torch.float32)
        slim_te._assign_leaf_tensor(blk, "norm.weight", w)
        slim_te._assign_leaf_tensor(blk, "scale_buf", buf)

        self.assertFalse(blk.norm.weight.is_meta)
        self.assertTrue(torch.equal(blk.norm.weight, w))
        self.assertFalse(blk.scale_buf.is_meta)
        self.assertTrue(torch.equal(blk.scale_buf, buf))
        # Sibling meta Linear must stay strictly on meta (never materialized onto CPU RAM).
        self.assertTrue(blk.proj.weight.is_meta)


if __name__ == "__main__":
    unittest.main()
