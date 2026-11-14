#!/usr/bin/env python3
"""Bit-exactness check: llama.cpp BQ2_0 quantizer vs. the KV1 training quantizer (kv_ternary.py).

    python tools/kv1/test_bq2_0_exact.py --ref path/to/kv_ternary.py --lib build/bin/libggml-base.so

kv_ternary.py ships with every KV1 model repo (e.g. klein-ml/KV1-1.2B-Prerelease).
Compares the fp16 scale d, the fp16 min m and the packed 2-bit codes byte for byte.
"""
import argparse, ctypes, importlib.util, sys
import numpy as np
import torch

GGML_TYPE_BQ2_0 = 42
QK = 64
BLOCK = 4 + QK // 4  # d, m (fp16) + 16 bytes of codes


def load_ref(path):
    spec = importlib.util.spec_from_file_location("kv_ternary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ggml_quantize(lib, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    nrows, n = x.shape
    out = np.zeros(nrows * (n // QK) * BLOCK, dtype=np.uint8)
    lib.ggml_quantize_chunk(GGML_TYPE_BQ2_0, x.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int64(0), ctypes.c_int64(nrows), ctypes.c_int64(n), None)
    return out.reshape(-1, BLOCK)


def ref_quantize(ref, x):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(-1, QK)
    codes, mn, sc = ref.quantize_ternary(t, mode="affine", eps=1e-6, levels=4)
    c = codes.to(torch.int32).numpy()
    packed = (c[:, 0::4] | (c[:, 1::4] << 2) | (c[:, 2::4] << 4) | (c[:, 3::4] << 6)).astype(np.uint8)
    # cross-check our packing against the reference pack_codes
    ref_packed = ref.pack_codes(codes).reshape(len(c), -1).cpu().numpy().astype(np.uint8)
    pack_ok = np.array_equal(ref_packed, packed)
    d = sc.reshape(-1).numpy().view(np.uint8).reshape(-1, 2)
    m = mn.reshape(-1).numpy().view(np.uint8).reshape(-1, 2)
    return np.concatenate([d, m, packed], axis=1), pack_ok


def cases(rng):
    n = 4096
    yield "normal", rng.standard_normal((n, QK)).astype(np.float32)
    yield "wide range", (rng.standard_normal((n, QK)) * 37.0 + 5.0).astype(np.float32)
    x = rng.standard_normal((n, QK)).astype(np.float32)
    x[:, rng.integers(0, QK, 3)] *= 40.0  # outlier channels, like raw attention keys
    yield "outliers", x
    H = torch.tensor(np.array([[1.0]]))
    while H.shape[0] < QK:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = (H / np.sqrt(QK)).float().numpy()
    yield "outliers, Hadamard-rotated", (x @ H).astype(np.float32)
    yield "constant rows", np.repeat(rng.standard_normal((n, 1)), QK, axis=1).astype(np.float32)
    yield "zeros", np.zeros((64, QK), np.float32)
    # exact .5 ties: min 0, max 3 -> d = 1 exactly in fp16, values k + 0.5
    t = rng.integers(0, 3, (n, QK)).astype(np.float32) + 0.5
    t[:, 0], t[:, 1] = 0.0, 3.0
    yield "ties at .5", t
    yield "tiny values", (rng.standard_normal((n, QK)) * 1e-7).astype(np.float32)
    yield "bf16-rounded", torch.from_numpy(rng.standard_normal((n, QK)).astype(np.float32)).bfloat16().float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lib", required=True)
    a = ap.parse_args()

    ref = load_ref(a.ref)
    lib = ctypes.CDLL(a.lib)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]

    rng = np.random.default_rng(0)
    ok_all = True
    for name, x in cases(rng):
        got = ggml_quantize(lib, x)
        want, pack_ok = ref_quantize(ref, x)
        same = got == want
        n_bad = int((~same.all(1)).sum())
        ok = n_bad == 0 and pack_ok
        ok_all &= ok
        print(f"{'OK  ' if ok else 'FAIL'} {name:28s} blocks {len(got):6d}  mismatched blocks {n_bad:5d}"
              f"  (d {int((~same[:, :2].all(1)).sum())}, m {int((~same[:, 2:4].all(1)).sum())}, codes {int((~same[:, 4:].all(1)).sum())})"
              f"{'' if pack_ok else '  PACKING ORDER DIFFERS FROM pack_codes'}")
    print("ALL BYTE-IDENTICAL" if ok_all else "MISMATCH")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
