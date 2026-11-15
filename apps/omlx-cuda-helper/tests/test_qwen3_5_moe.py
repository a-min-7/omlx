# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper import checkpoint, families, tensors  # noqa: E402
from omlx_cuda_helper.families import qwen3_5_moe  # noqa: E402
from omlx_cuda_helper.families.qwen3_5_moe.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/qwen3_5_moe"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)


def _case():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {name: (data[name], dtype) for name, dtype, _, _ in meta["manifest"]}
    return meta, arrays


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


def _bounds():
    """The cuts to replay, sized from the fixture's own layer counts."""
    if not FIXTURE.exists():
        return [(0, 5)]  # never used: the module is skipped without the fixture
    meta = json.loads((FIXTURE / "meta.json").read_text())
    exercised, held = meta["layers"], meta["model_layers"]
    cuts = {(0, exercised), (0, held)}
    cuts |= {(0, cut, exercised) for cut in range(1, exercised)}
    return sorted(cuts)


def _replay(bounds, capture=None):
    """The fixture's chunks through stages cut at ``bounds``.

    Yields each chunk's index, the boundary at the end of every stage (by
    its last layer), the state of the whole chain and what the stages
    captured for ``capture``. Only the first stage gets the host's inputs.
    """
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == qwen3_5_moe.CONTRACT
    assert all(a >= qwen3_5_moe.FIRST_STAGE_MIN for a in bounds[1:-1])
    stages = [
        load_stage(FIXTURE / "model", first, last)
        for first, last in zip(bounds, bounds[1:])
    ]
    for stage in stages:
        # A layer loaded again after a long prompt runs as at the first load.
        stage.drop_layer()
        stage.add_layer()
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage, job in zip(stages, jobs):
        # On CUDA the replay then runs through the self-checked kernels.
        stage.enable_kernels()
        stage.load_prefix(job, _group(arrays, "p:"))
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start = 0
    for index, size in enumerate(sizes[skip:]):
        boundary, cuts, state, captured = None, {}, {}, {}
        for stage, job in zip(stages, jobs):
            inputs = {} if boundary else _group(arrays, f"in{index}:")
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs=inputs,
                following=0,
                capture=capture,
            )
            assert all(stage.first <= int(n.split(".")[0]) < stage.last for n in own)
            state.update(own)
            captured.update({n: boundary.pop(n) for n in list(boundary) if "." in n})
            cuts[min(stage.last, k)] = boundary
        start += size
        yield index, cuts, state, captured


@pytest.mark.parametrize(
    "bounds", _bounds(), ids=lambda bounds: "cut-" + "-".join(map(str, bounds))
)
def test_stages_reproduce_the_contract_fixture(bounds):
    meta, arrays = _case()
    for index, cuts, state, _ in _replay(bounds):
        boundary = cuts[meta["layers"]]
        for tag, got in ((f"b{index}:", boundary), (f"s{index}:", state)):
            want = _group(arrays, tag)
            assert sorted(got) == sorted(want), tag
            for name, (data, dtype) in got.items():
                expected, expected_dtype = want[name]
                assert (dtype, data.dtype, data.shape) == (
                    expected_dtype,
                    expected.dtype,
                    expected.shape,
                ), (
                    tag + name
                )
                np.testing.assert_allclose(
                    data,
                    expected,
                    rtol=meta["rtol"],
                    atol=meta["atol"],
                    err_msg=tag + name,
                )


@pytest.mark.skipif(not mx.cuda.is_available(), reason="a GPU load runs on CUDA")
def test_weights_loaded_into_the_gpu_pool_give_the_same_stages(monkeypatch):
    monkeypatch.setattr(tensors, "weights_on_gpu", True)
    k = _case()[0]["layers"]
    test_stages_reproduce_the_contract_fixture((0, 2, k) if k > 2 else (0, k))


def test_a_stage_returns_the_layers_a_block_drafter_reads():
    meta, _ = _case()
    k, skip = meta["layers"], meta["prefix_chunks"]
    rest = meta["sizes"][skip:]
    if k < 3 or len(rest) < 2:
        pytest.skip("the fixture needs a second chunk and three layers to capture")
    # The window starts two rows into the second replayed chunk.
    window = rest[0] + 2
    # Layer k is past the job's layers, so no stage captures it.
    layers, job = [1, k - 1, k], [1, k - 1]
    chunks = list(_replay((0, 2, k), {"layers": layers, "from": window}))
    assert [sorted(c[3]) for c in chunks] == [[], sorted(f"cap.{i}" for i in job)]
    _, cuts, _, captured = chunks[1]
    for layer in job:
        got, dtype = captured[f"cap.{layer}"]
        want, want_dtype = cuts[layer + 1]["h"]
        assert dtype == want_dtype
        np.testing.assert_array_equal(got, want[:, window - rest[0] :])


def test_a_repeated_kernel_self_check_still_compares_with_plain_ops(monkeypatch):
    config = json.loads((FIXTURE / "model" / "config.json").read_text())
    linear = qwen3_5_moe.linear_layers(qwen3_5_moe.text_config(config))
    if False not in linear:
        pytest.skip("the fixture has no full-attention layer")
    index = linear.index(False)
    attention = load_stage(FIXTURE / "model", index, index + 1).layers[0].mixer

    def wrong(queries, keys, values, scale, mask):
        return mx.ones_like(queries)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", wrong)
    # As an earlier self-check had left fused attention in use.
    attention.fused = True
    attention.enable_kernels()
    assert attention.fused is False


def test_plan_counts_only_what_a_stage_loads():
    meta, _ = _case()
    assert families.contracts()["qwen3_5_moe"] == qwen3_5_moe.CONTRACT
    shared, layers = qwen3_5_moe.plan(FIXTURE / "model")
    assert len(layers) == meta["model_layers"] and min(layers) > 0
    # The embedding is shared; the output head and the final norm stay on
    # the host.
    assert shared > 0
    assert not qwen3_5_moe.resident("language_model.mtp.layers.0.mlp.up_proj.weight")
    # A Hugging Face checkpoint needs the host's conversion first.
    with pytest.raises(ValueError, match="MLX layout"):
        qwen3_5_moe.check_layout(
            {"model.language_model.layers.0.mlp.up_proj.weight": 0}
        )


def test_plan_leaves_the_3d_expert_banks_packed(monkeypatch):
    """A bank unpacked as a 2-D linear would inflate the per-layer estimate.

    The recorded fixture is unquantized, so this gives its banks the packed
    tensors a quantized checkpoint carries: the plan must still skip them on
    their 3-D shape, not count them as unpacked linears.
    """
    model = FIXTURE / "model"
    index = checkpoint.tensors(model)
    banks = {
        name: tensor
        for name, tensor in index.items()
        if name.endswith(".weight")
        and ".switch_mlp." in name
        and len(tensor.shape) == 3
    }
    if not banks:
        pytest.skip("the fixture has no 3-D expert banks")
    config = {
        **checkpoint.load_config(model),
        "quantization": {"group_size": 64, "bits": 4},
    }
    packed = dict(index)
    for name, tensor in banks.items():
        for part in ("scales", "biases"):
            packed[f"{name.removesuffix('.weight')}.{part}"] = replace(
                tensor, shape=(tensor.shape[0], 1)
            )
    monkeypatch.setattr(checkpoint, "load_config", lambda _: config)
    monkeypatch.setattr(checkpoint, "tensors", lambda _: packed)
    _, kept = qwen3_5_moe.plan(model)
    inflated = dict(packed)
    for name, tensor in banks.items():
        inflated[name] = replace(
            tensor, shape=(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
        )
    monkeypatch.setattr(checkpoint, "tensors", lambda _: inflated)
    _, unpacked = qwen3_5_moe.plan(model)
    assert len(unpacked) == len(kept)
    assert max(unpacked) > max(kept)
