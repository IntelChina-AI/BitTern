"""Pack CAT-Q ternary codes and scales into GGML `Q2_0` blocks.

`Q2_0` is the group-128 ternary weight type used by the Bonsai llama.cpp fork
(https://github.com/PrismML-Eng/llama.cpp, branch `prism`).  The layout below
is taken from that source tree, not from documentation:

  ggml/src/ggml-common.h
      #define QK2_0 128
      typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;

  ggml/src/ggml-quants.c :: dequantize_row_q2_0
      byte_index = j / 4;  bit_offset = (j % 4) * 2
      q = (qs[byte_index] >> bit_offset) & 0x03
      y[j] = (q - 1) * d          ->  00: -1   01: 0   10: +1   11: +2

A block is 34 bytes per 128 weights (2.125 bits per weight).  A CAT-Q group
therefore maps with `code -> code + 1` and `d = fp16(scale)`; the `+2` level is
never produced.  GGML rows run along `ne[0] = in_features`, the same direction
CAT-Q groups over, so no transpose or regrouping is needed.
"""

import numpy as np

QK2_0 = 128
BLOCK_BYTES = 2 + QK2_0 // 4  # fp16 scale + 32 packed bytes

# ggml/include/ggml.h :: GGML_TYPE_Q2_0 = 42
GGML_TYPE_Q2_0 = 42
# src/llama-model.cpp / include/llama.h :: LLAMA_FTYPE_MOSTLY_Q2_0 = 41
LLAMA_FTYPE_MOSTLY_Q2_0 = 41


def register_q2_0():
    """Teach the `gguf` package about `Q2_0`.

    The upstream PyPI package stops at type 41, so type 42 is free.  Only the
    block geometry is needed: the writer uses it to recover the logical tensor
    shape from the packed byte shape.
    """
    from gguf.constants import GGML_QUANT_SIZES

    GGML_QUANT_SIZES.setdefault(GGML_TYPE_Q2_0, (QK2_0, BLOCK_BYTES))
    return GGML_TYPE_Q2_0


def pack_q2_0(codes, scales):
    """codes: (..., out, in) int8 in {-1, 0, +1}; scales: one float per group.

    Returns a uint8 array of shape (..., out, in // 128 * 34).  Leading
    dimensions (used by stacked MoE expert tensors, shape (n_expert, out, in))
    pass through untouched: GGML stores those as plain row-major rows as well.
    """
    codes = np.asarray(codes)
    if codes.ndim < 2:
        raise ValueError(f"expected at least a 2-D weight, got shape {codes.shape}")
    lead_shape = tuple(codes.shape[:-2])
    out_features, in_features = codes.shape[-2:]
    if in_features % QK2_0 != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of {QK2_0}")

    n_blocks = int(np.prod(codes.shape)) // QK2_0
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    if scales.size != n_blocks:
        raise ValueError(f"expected {n_blocks} scales, got {scales.size}")

    quants = codes.astype(np.int8).reshape(n_blocks, QK2_0)
    if quants.min() < -1 or quants.max() > 1:
        raise ValueError("ternary codes must lie in {-1, 0, +1}")

    levels = (quants + np.int8(1)).astype(np.uint8).reshape(n_blocks, QK2_0 // 4, 4)
    shifted = levels << np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4)
    qs = shifted[..., 0] | shifted[..., 1] | shifted[..., 2] | shifted[..., 3]

    d = scales.astype(np.float16).reshape(n_blocks, 1).view(np.uint8)

    blocks = np.concatenate([d, qs], axis=-1)
    assert blocks.shape == (n_blocks, BLOCK_BYTES)
    return blocks.reshape(*lead_shape, out_features, in_features // QK2_0 * BLOCK_BYTES)


def unpack_q2_0(blocks, shape):
    """Inverse of `pack_q2_0`, returning float32 weights of `shape`."""
    raw = np.ascontiguousarray(blocks).reshape(-1, BLOCK_BYTES)
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)
    qs = raw[:, 2:]
    shifted = qs.reshape(qs.shape[0], -1, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4)
    codes = (shifted & 0x03).reshape(qs.shape[0], QK2_0).astype(np.int8) - np.int8(1)
    return (codes.astype(np.float32) * d).reshape(shape)
