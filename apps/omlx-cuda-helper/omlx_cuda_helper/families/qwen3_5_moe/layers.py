# SPDX-License-Identifier: Apache-2.0
"""Prefill forward of the Qwen3.5-MoE blocks, batch one and text only.

A copy of the dense qwen3_5 forward with the dense MLP replaced by the routed
expert block, so attention, GatedDeltaNet, KVCache and the quantization seams
stay byte-identical to the family this derives from.

Each block follows the path the oMLX host takes for a prefill chunk, op by op
and in the same dtypes, so 16-bit activations round as they do there. The
blocks are ports of the mlx-vlm classes (MIT, Copyright (c) 2025 Prince
Canuma); the attention follows the host's fused kernel, which keeps scores,
softmax and value sums in float32.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ... import checkpoint
from ...tensors import Linear
from . import kernels

# Bytes of the float32 score sheet one attention step holds.
_SHEET_BYTES = 1 << 30


class Weights:
    """Lazy checkpoint tensors under one name prefix, with their quantization."""

    def __init__(self, arrays: dict[str, mx.array], config: dict[str, Any], root: str):
        self.arrays, self.config, self.root = arrays, config, root

    def at(self, path: str) -> Weights:
        return Weights(self.arrays, self.config, f"{self.root}{path}.")

    def __getitem__(self, name: str) -> mx.array:
        return self.arrays[self.root + name]

    def __contains__(self, name: str) -> bool:
        return self.root + name in self.arrays

    def quantization(self, name: str) -> dict[str, Any] | None:
        return checkpoint.quantization(self.config, self.root + name)

    def linear(self, name: str) -> Linear:
        """A dense linear layer in the dtype the checkpoint computes in."""
        path = self.root + name
        return Linear(self.arrays, path, checkpoint.quantization(self.config, path))


def causal_conv(x: mx.array, taps: mx.array) -> mx.array:
    """Depthwise conv of [1, rows + K - 1, C] with float32 ``taps`` [C, K].

    The taps add up in float32 in order and round once, like the host conv.
    """
    rows = x.shape[1] - taps.shape[1] + 1
    acc = x[:, :rows].astype(mx.float32) * taps[:, 0]
    for i in range(1, taps.shape[1]):
        acc = acc + x[:, i : i + rows].astype(mx.float32) * taps[:, i]
    return acc.astype(x.dtype)


class GatedDeltaNet:
    """Linear attention: a causal conv and the gated delta recurrence."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.key_heads = text["linear_num_key_heads"]
        self.value_heads = text["linear_num_value_heads"]
        self.key_dim = text["linear_key_head_dim"]
        self.value_dim = text["linear_value_head_dim"]
        self.eps = text["rms_norm_eps"]
        self.qkv = w.linear("in_proj_qkv")
        self.z = w.linear("in_proj_z")
        self.b = w.linear("in_proj_b")
        self.a = w.linear("in_proj_a")
        self.out = w.linear("out_proj")
        self.taps = w["conv1d.weight"][..., 0].astype(mx.float32)
        self.decay = mx.exp(w["A_log"].astype(mx.float32))
        self.dt_bias = w["dt_bias"]
        self.norm_weight = w["norm.weight"]
        self.conv = causal_conv

    def enable_kernels(self) -> None:
        """Use the conv kernel if it equals ``causal_conv`` on these taps."""
        rows = 64 + self.taps.shape[1] - 1
        x = mx.random.normal((1, rows, self.taps.shape[0]))
        x = x.astype(self.qkv.weight.dtype)
        got = kernels.causal_conv(x, self.taps)
        if mx.array_equal(got, causal_conv(x, self.taps)).item():
            self.conv = kernels.causal_conv
        else:
            kernels.logger.warning("Causal conv kernel differs; using the reference")

    def parameters(self) -> list[mx.array]:
        linears = (self.qkv, self.z, self.b, self.a, self.out)
        own = [self.taps, self.decay, self.dt_bias, self.norm_weight]
        return own + [a for layer in linears for a in layer.parameters()]

    def prework(self, qkv: mx.array, window: mx.array | None) -> tuple[mx.array, ...]:
        """(q, k, v, next conv window) from the projected rows [1, rows, C]."""
        rows = qkv.shape[1]
        if window is None:
            width = self.taps.shape[1] - 1
            window = mx.zeros((1, width, qkv.shape[-1]), dtype=qkv.dtype)
        conv_in = mx.concatenate([window, qkv], axis=1)
        conv_out = nn.silu(self.conv(conv_in, self.taps))
        keys = self.key_heads * self.key_dim
        q = conv_out[..., :keys].reshape(1, rows, self.key_heads, self.key_dim)
        k = conv_out[..., keys : 2 * keys].reshape(q.shape)
        v = conv_out[..., 2 * keys :]
        v = v.reshape(1, rows, self.value_heads, self.value_dim)
        # L2 norm as an RMS norm; the query also takes the 1/sqrt(dim) of
        # the readout. Both scales multiply the rounded norm.
        scale = self.key_dim**-0.5
        eps = 1e-6 * scale**2
        q = (scale**2) * mx.fast.rms_norm(q, None, eps)
        k = scale * mx.fast.rms_norm(k, None, eps)
        return q, k, v, conv_in[:, rows:]

    def gates(self, a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
        """(decay in float32, update rate) of each step and value head."""
        return mx.exp(-self.decay * nn.softplus(a + self.dt_bias)), mx.sigmoid(b)

    def finish(self, y: mx.array, z: mx.array) -> mx.array:
        """The recurrence output normalized and gated, [1, rows, value width]."""
        y = mx.fast.rms_norm(y, self.norm_weight, self.eps)
        gate = nn.silu(z.reshape(y.shape).astype(mx.float32))
        gated = (gate * y.astype(mx.float32)).astype(y.dtype)
        return gated.reshape(1, y.shape[1], -1)

    def __call__(self, x: mx.array, cache: list) -> mx.array:
        q, k, v, cache[0] = self.prework(self.qkv(x), cache[0])
        gamma, beta = self.gates(self.a(x), self.b(x))
        y, cache[1] = mx.fast.gated_delta_update(q, k, v, gamma, beta, cache[1])
        return self.out(self.finish(y, self.z(x)))


class KVCache:
    """Rotated keys and values of one attention layer for one prompt."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        # (keys, values) of the last chunk.
        self.appended: tuple[mx.array, ...] = ()

    def append(self, keys: mx.array, values: mx.array) -> None:
        """Add rows [1, kv heads, n, dim]."""
        self.appended = (keys, values)
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]

    def arrays(self) -> list[mx.array]:
        return [a for a in (self.keys, self.values) if a is not None]


def rotary_frequencies(text: dict[str, Any]) -> mx.array:
    """Inverse frequencies of the rotary dims as this backend rounds them.

    Only a warm-up chunk uses them: a prompt's chunks carry the frequencies
    of the host, whose last bits differ.
    """
    rope = text.get("rope_parameters") or {}
    heads = text["num_attention_heads"]
    head_dim = text.get("head_dim") or text["hidden_size"] // heads
    rotary = int(head_dim * rope.get("partial_rotary_factor", 0.25))
    exponent = mx.arange(0, rotary, 2).astype(mx.float32) / rotary
    return 1.0 / (rope.get("rope_theta", 10_000_000) ** exponent)


def rotate(x: mx.array, positions: mx.array, frequencies: mx.array) -> mx.array:
    """Rotate the leading dims of [1, heads, n, dim] by text positions [n].

    Angles, sines and the rotation are float32 and the result rounds once.
    """
    half = frequencies.shape[0]
    angle = positions.astype(mx.float32)[:, None] * frequencies
    cos, sin = mx.cos(angle), mx.sin(angle)
    a = x[..., :half].astype(mx.float32)
    b = x[..., half : 2 * half].astype(mx.float32)
    parts = [(a * cos - b * sin).astype(x.dtype), (b * cos + a * sin).astype(x.dtype)]
    if x.shape[-1] > 2 * half:
        parts.append(x[..., 2 * half :])
    return mx.concatenate(parts, axis=-1)


class Attention:
    """Gated causal attention over every cached token."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.heads = text["num_attention_heads"]
        self.kv_heads = text["num_key_value_heads"]
        self.head_dim = text.get("head_dim") or text["hidden_size"] // self.heads
        self.eps = text["rms_norm_eps"]
        self.q_proj = w.linear("q_proj")
        self.k_proj = w.linear("k_proj")
        self.v_proj = w.linear("v_proj")
        self.o_proj = w.linear("o_proj")
        self.q_norm = w["q_norm.weight"]
        self.k_norm = w["k_norm.weight"]
        self.fused = False

    def parameters(self) -> list[mx.array]:
        linears = (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        own = [self.q_norm, self.k_norm]
        return own + [a for layer in linears for a in layer.parameters()]

    def enable_kernels(self) -> None:
        """Use fused attention if it matches ``attend`` on plain ops."""
        dtype = self.q_proj.weight.dtype
        rows, total = 40, 104
        keys = mx.random.normal((1, self.kv_heads, total, self.head_dim)).astype(dtype)
        values = mx.random.normal(keys.shape).astype(dtype)
        queries = mx.random.normal((1, self.heads, rows, self.head_dim)).astype(dtype)
        # The reference runs on plain ops, also when fused attention is in use.
        self.fused = False
        want = self.attend(queries, keys, values, step=16)
        self.fused = True
        if not kernels.attention_matches(self.attend(queries, keys, values), want):
            self.fused = False

    def attend(
        self, queries: mx.array, keys: mx.array, values: mx.array, step: int = 0
    ) -> mx.array:
        """Causal attention of the last rows of the cache over all of it.

        ``queries`` [1, heads, rows, dim] are the newest ``rows`` tokens of
        ``keys`` / ``values`` [1, kv heads, total, dim]. Returns
        [1, rows, heads, dim]. Without fused attention, scores, softmax and
        value sums are float32 and the output rounds once; ``step`` query
        rows then share a score sheet, and 0 sizes it to ``_SHEET_BYTES``.
        """
        rows, total = queries.shape[2], keys.shape[2]
        past = total - rows
        kv, dim = self.kv_heads, self.head_dim
        groups = self.heads // kv
        # Exact for a head width that is a power of four; else it moves the
        # float32 scores by less than their own rounding.
        scale = 1.0 / math.sqrt(dim)
        if self.fused:
            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask="causal"
            )
            return out.transpose(0, 2, 1, 3)
        grouped = queries.reshape(1, kv, groups, rows, dim)
        # Transposed once: every step multiplies by the same keys.
        keys = mx.contiguous(keys.astype(mx.float32).swapaxes(-1, -2))
        values = values.astype(mx.float32)
        step = step or max(1, _SHEET_BYTES // (4 * self.heads * total))
        outputs = []
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            n = stop - start
            q = grouped[..., start:stop, :].astype(mx.float32) * scale
            scores = q.reshape(1, kv, groups * n, dim) @ keys
            visible = past + mx.arange(start, stop)[:, None] >= mx.arange(total)
            scores = scores.reshape(1, kv, groups, n, total)
            weights = mx.softmax(mx.where(visible, scores, -mx.inf), axis=-1)
            out = weights.reshape(1, kv, groups * n, total) @ values
            outputs.append(out.astype(queries.dtype).reshape(1, kv, groups, n, dim))
            # One step at a time: each holds a float32 sheet of its scores.
            mx.eval(outputs[-1])
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=-2)
        return out.reshape(1, self.heads, rows, dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, cache: KVCache, frequencies: mx.array) -> mx.array:
        rows = x.shape[1]
        projected = self.q_proj(x).reshape(1, rows, self.heads, 2 * self.head_dim)
        queries, gate = projected[..., : self.head_dim], projected[..., self.head_dim :]
        keys = self.k_proj(x).reshape(1, rows, self.kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(1, rows, self.kv_heads, self.head_dim)
        queries = mx.fast.rms_norm(queries, self.q_norm, self.eps).transpose(0, 2, 1, 3)
        keys = mx.fast.rms_norm(keys, self.k_norm, self.eps).transpose(0, 2, 1, 3)
        positions = mx.arange(cache.offset, cache.offset + rows, dtype=mx.int32)
        queries = rotate(queries, positions, frequencies)
        cache.append(rotate(keys, positions, frequencies), values.transpose(0, 2, 1, 3))
        output = self.attend(queries, cache.keys, cache.values)
        gate = mx.sigmoid(gate.reshape(1, rows, -1))
        return self.o_proj(output.reshape(1, rows, -1) * gate)


# --- copied from qwen4_exp so this file is self-contained (no cross-family import)

def stacked_sum(x: mx.array) -> mx.array:
    """``x.sum(-2)`` over fewer than 32 rows, rounded as on the host.

    MLX's Metal reduce adds the rows eight apart first and then those eight
    partial sums in order, all in the input dtype.
    """
    rows = x.shape[-2]
    if x.dtype == mx.float32 or rows >= 32:
        return x.sum(axis=-2)
    partial = []
    for first in range(min(rows, 8)):
        total = x[..., first, :]
        for row in range(first + 8, rows, 8):
            total = x[..., row, :] + total
        partial.append(total)
    total = partial[0]
    for part in partial[1:]:
        total = part + total
    return total


class MoE:
    """Routed expert MLPs plus a gated shared expert."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.top_k = text["num_experts_per_tok"]
        self.inner = text["moe_intermediate_size"]
        self.normalize = text.get("norm_topk_prob", True)
        self.router = w.linear("gate")
        self.shared_gate = w.linear("shared_expert.gate_proj")
        self.shared_up = w.linear("shared_expert.up_proj")
        self.shared_down = w.linear("shared_expert.down_proj")
        self.shared_scale = w.linear("shared_expert_gate")
        banks = w.at("switch_mlp")
        # Gate and up share their input rows: one gather computes both.
        self.gate_up = self._bank(banks, ("gate_proj", "up_proj"))
        self.down = self._bank(banks, ("down_proj",))
        self.combine = None
        self.gather = None
        self.swiglu = None
        self.expert_kernel = False

    @staticmethod
    def _bank(w: Weights, names: tuple[str, ...]) -> tuple:
        """(weight, scales, biases, quantization) of expert projections.

        Several projections of the same input are joined along their outputs.
        """
        if f"{names[0]}.scales" not in w:
            # Dense banks are stored [experts, out, in]; gather_mm wants in first.
            parts = [w[f"{name}.weight"].swapaxes(-1, -2) for name in names]
            return mx.concatenate(parts, axis=-1), None, None, None
        joined = [
            mx.concatenate([w[f"{name}.{part}"] for name in names], axis=1)
            for part in ("weight", "scales", "biases")
        ]
        return *joined, w.quantization(names[0])

    def parameters(self) -> list[mx.array]:
        linears = (
            self.router,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
            self.shared_scale,
        )
        banks = [a for bank in (self.gate_up, self.down) for a in bank[:3]]
        return [a for a in banks if a is not None] + [
            a for layer in linears for a in layer.parameters()
        ]

    def enable_kernels(self) -> None:
        """Pick the CUDA expert kernels this block may use, or none.

        Two independent reasons to use none, and the first is why this family
        never does today:

        * the expert banks are PACKED quantized tuples ``(uint32, scales,
          biases, quant)``. The expert kernels were written and validated
          against qwen4_exp's DENSE banks ``(weight, None, None, None)``; a
          kernel that read packed uint32 as numeric data would still produce a
          speedup and still diverge under a negative control while returning
          garbage on every long chunk. Disabled until a quantized kernel exists.
        * this family's ``kernels`` module is the dense one and carries no
          expert kernel at all, so the probes below would raise AttributeError.

        The reference paths (``rows[order // k]``, ``mx.gather_qmm``,
        ``_swiglu``, ``stacked_sum``) are the correctness baseline, not a
        degraded fallback.
        """
        self._log_dtypes()
        quantized = self.gate_up[3] is not None or self.down[3] is not None
        if quantized:
            kernels.logger.info(
                "MoE expert kernels DISABLED: expert banks are quantized "
                "(no quantized expert kernel exists); using gather_qmm/reference"
            )
            return
        if not hasattr(kernels, "combine_experts"):
            kernels.logger.info(
                "MoE expert kernels DISABLED: kernels module has no expert kernels"
            )
            return
        k, dtype = self.top_k, self.router.weight.dtype
        y = mx.random.normal((64 * k, 1, self.router.weight.shape[1])).astype(dtype)
        positions = mx.argsort(mx.random.uniform(shape=(64 * k,)))
        scores = mx.random.uniform(shape=(64, k)).astype(dtype)
        want = stacked_sum(y[positions].reshape(64, k, -1) * scores[..., None])
        if kernels.close(
            kernels.combine_experts(y, positions, scores), want, "MoE sum"
        ):
            self.combine = kernels.combine_experts
        x = mx.random.normal((64, self.router.weight.shape[1])).astype(dtype)
        src = mx.random.randint(0, 64, (64 * k,)).astype(mx.uint32)
        if x.shape[1] * dtype.size % 16:
            return
        if mx.array_equal(kernels.gather_rows(x, src), x[src]).item():
            self.gather = kernels.gather_rows
        else:
            kernels.logger.warning("Row gather kernel differs; using the reference")
        both = mx.random.normal((64, 1, 2 * self.inner)).astype(dtype)
        if dtype == mx.bfloat16 and self.inner % 8 == 0:
            if mx.array_equal(kernels.swiglu(both), self._swiglu(both)).item():
                self.swiglu = kernels.swiglu
            else:
                kernels.logger.warning("SwiGLU kernel differs; using the reference")
        widths = ((self.gate_up, x.shape[1]), (self.down, self.inner))
        if (
            dtype == mx.bfloat16
            and self.swiglu is not None
            and all(kernels.expert_gemm_fits(bank, width) for bank, width in widths)
        ):
            self.expert_kernel = self._experts_match(dtype)

    def _log_dtypes(self) -> None:
        """Log the router, gate and output dtypes once per layer at load.

        P0-1: the host router is bf16 and ``softmax(precise=True)`` returns the
        input dtype, so top-k is taken over bf16-rounded probabilities. If this
        path produced fp32 it would both change expert choice on near-ties and
        promote the residual stream for every downstream layer and the handoff
        tensor. Measured here rather than assumed.
        """
        w = self.router.weight
        probe = mx.random.normal((1, 8, w.shape[1])).astype(w.dtype)
        logits = self.router(probe)
        gates = mx.softmax(logits, axis=-1, precise=True)
        out = self(probe)
        mx.eval(gates, out)
        kernels.logger.info(
            "MoE dtypes: router_weight=%s logits=%s gates=%s output=%s output_dtype_is_bf16=%s",
            w.dtype, logits.dtype, gates.dtype, out.dtype, str(out.dtype) == "bfloat16",
        )

    def _experts_match(self, dtype: mx.Dtype) -> bool:
        """Whether the expert kernel path gives the bits of gather_qmm.

        Enough routes for the sorted gather, and one expert with more rows
        than a launch block holds at once.
        """
        tokens, count = 256, self.gate_up[0].shape[0]
        x = mx.random.normal((1, tokens, self.router.weight.shape[1])).astype(dtype)
        others = mx.random.randint(1, count, (1, tokens, self.top_k - 1))
        chosen = mx.concatenate([mx.zeros((1, tokens, 1), mx.int32), others], -1)
        self.expert_kernel = False
        want = self.experts(x, chosen)[0]
        self.expert_kernel = True
        if mx.array_equal(self.experts(x, chosen)[0], want).item():
            return True
        kernels.logger.warning("Expert kernel differs; using gather_qmm")
        return False

    @staticmethod
    def _gather(x: mx.array, bank: tuple, experts: mx.array) -> mx.array:
        """Rows [n, 1, in] through each row's expert; ``experts`` is sorted."""
        weight, scales, biases, quant = bank
        if quant is None:
            return mx.gather_mm(x, weight, rhs_indices=experts, sorted_indices=True)
        return mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            rhs_indices=experts,
            transpose=True,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            sorted_indices=True,
        )

    @staticmethod
    def _swiglu(both: mx.array) -> mx.array:
        inner = both.shape[-1] // 2
        return nn.silu(both[..., :inner]) * both[..., inner:]

    def route(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        """(expert ids, weights) [1, rows, top_k] from the router logits."""
        k = self.top_k
        gates = mx.softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, chosen, axis=-1)
        if self.normalize:
            # The host's op, verbatim: mlx_lm Qwen3NextSparseMoeBlock does
            #   scores = scores / scores.sum(axis=-1, keepdims=True)
            # qwen4_exp's family keeps a sequential variant because ITS host adds
            # the k values one at a time; this family deliberately does not, so
            # the divisor is the identical reduction the host runs. With k=8 in
            # bf16 a tree reduce and a sequential chain can differ by an ulp.
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return chosen, scores

    def experts(self, x: mx.array, chosen: mx.array) -> tuple[mx.array, mx.array]:
        """Expert outputs [rows * top_k, 1, hidden] sorted by expert, and the
        row of each (token, choice) in them."""
        assert x.shape[0] == 1, f"the expert path assumes batch one, got {x.shape}"
        k = self.top_k
        flat = chosen.reshape(-1)
        order = mx.argsort(flat)
        ids = flat[order].astype(mx.uint32)
        rows = x.reshape(x.shape[1], -1)
        # gather_qmm takes its sorted expert kernel, whose sums this kernel
        # reproduces, from four routes an expert on.
        if self.expert_kernel and ids.shape[0] >= 4 * self.gate_up[0].shape[0]:
            # Each route reads its token's row; gate, up and SwiGLU in one launch.
            src = (order // k).astype(mx.uint32)
            offsets = kernels.expert_offsets(ids, self.gate_up[0].shape[0])
            hidden = kernels.expert_gemm(rows, self.gate_up, offsets, src, swiglu=True)
            y = kernels.expert_gemm(hidden, self.down, offsets)
            return y[:, None], mx.argsort(order)
        if self.gather is not None:
            routed = self.gather(rows, (order // k).astype(mx.uint32))[:, None]
        else:
            routed = rows[order // k][:, None]
        both = self._gather(routed, self.gate_up, ids)
        hidden = (self.swiglu or self._swiglu)(both)
        return self._gather(hidden, self.down, ids), mx.argsort(order)

    def combined(self, y: mx.array, positions: mx.array, scores: mx.array) -> mx.array:
        """Each token's expert outputs times their weights, summed."""
        if self.combine is not None:
            return self.combine(y, positions, scores[0])[None]
        rows, k = scores.shape[1:]
        return stacked_sum(y[positions].reshape(1, rows, k, -1) * scores[..., None])

    def shared(self, x: mx.array) -> mx.array:
        shared = self.shared_down(nn.silu(self.shared_gate(x)) * self.shared_up(x))
        return mx.sigmoid(self.shared_scale(x)) * shared

    def __call__(self, x: mx.array) -> mx.array:
        chosen, scores = self.route(self.router(x))
        return self.combined(*self.experts(x, chosen), scores) + self.shared(x)


class Layer:
    """One decoder layer: linear or full attention, then the MLP."""

    def __init__(self, w: Weights, text: dict[str, Any], linear: bool):
        self.linear = linear
        self.eps = text["rms_norm_eps"]
        self.input_norm = w["input_layernorm.weight"]
        self.post_norm = w["post_attention_layernorm.weight"]
        if linear:
            self.mixer = GatedDeltaNet(w.at("linear_attn"), text)
        else:
            self.mixer = Attention(w.at("self_attn"), text)
        # Rooted at "mlp." so the expert banks resolve as mlp.switch_mlp.*,
        # the routers as mlp.gate and the shared expert as mlp.shared_expert.*.
        self.moe = MoE(w.at("mlp"), text)

    def parameters(self) -> list[mx.array]:
        own = [self.input_norm, self.post_norm, *self.mixer.parameters()]
        return own + self.moe.parameters()

    def enable_kernels(self) -> None:
        """Switch every block to the CUDA kernels that pass their self-check.

        Both blocks count: the mixer's attention/GDN kernels, and the MoE's
        combine_experts / gather_rows / swiglu / expert_gemm. qwen4_exp iterates
        its blocks the same way; enabling only the mixer would leave every expert
        op on its reference path, which is correct but silently slower — and on
        this arm speed is the measurement.
        """
        for block in (self.mixer, self.moe):
            block.enable_kernels()

    def new_cache(self) -> Any:
        return [None, None] if self.linear else KVCache()

    def __call__(
        self, x: mx.array, cache: Any, frequencies: mx.array | None
    ) -> mx.array:
        normed = mx.fast.rms_norm(x, self.input_norm, self.eps)
        if self.linear:
            h = x + self.mixer(normed, cache)
        else:
            h = x + self.mixer(normed, cache, frequencies)
        return h + self.moe(mx.fast.rms_norm(h, self.post_norm, self.eps))
