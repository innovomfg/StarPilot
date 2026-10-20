#!/usr/bin/env python3
"""Retarget only the external warped-image buffer, never GPU code or weights."""
import argparse
from dataclasses import replace
import faulthandler
import os
import sys
from pathlib import Path


def retarget_warped_input(jit, target):
    from tinygrad.engine.jit import CapturedJit
    from tinygrad.uop.ops import Ops
    captured = jit.captured
    assert captured is not None
    names = captured.expected_names
    slot = names.index("warped")
    info = captured.expected_input_info
    assert len(info) == len(names)
    assert info[slot][-1] == "CPU"
    assert sum(entry[-1] == "CPU" for entry in info) == 1
    nodes = list(captured._linear.toposort())
    programs = {u for u in nodes if u.op is Ops.PROGRAM}
    assert programs and all(u.arg.target.device == "AMD" and u.arg.target.arch == "gfx1102" for u in programs)
    weights = {u for u in nodes if u.op is Ops.BUFFER}
    assert all(u.device != "CPU" for u in weights), "Unexpected stored CPU buffer"
    source_params = [u for u in nodes if u.op is Ops.PARAM and u.arg.device == "CPU"]
    root_params = [u for u in captured._linear.toposort(enter_calls=False)
                   if u.op is Ops.PARAM and u.arg.device == "CPU"]
    print("SOURCE_PARAMETER_LAYOUT", [(u.arg.slot, u.arg.size, str(u.dtype)) for u in source_params],
          "ROOT_SLOT", slot, flush=True)
    assert root_params and all(u.arg.slot == slot for u in root_params)
    # Nested copy/graph functions number parameters locally, not by outer JIT slot.
    assert source_params and all(u.arg.size == root_params[0].arg.size and u.dtype == root_params[0].dtype
                                 for u in source_params)
    substitutions = {u: u.replace(arg=replace(u.arg, device=target)) for u in source_params}
    linear = captured._linear.substitute(substitutions, walk=True, enter_calls=True)
    new_nodes = list(linear.toposort())
    assert {u for u in new_nodes if u.op is Ops.PROGRAM} == programs, "GPU programs changed"
    assert {u for u in new_nodes if u.op is Ops.BUFFER} == weights, "Model buffers changed"
    assert not any(u.op is Ops.PARAM and u.arg.device == "CPU" for u in new_nodes)
    new_info = list(info)
    new_info[slot] = (*info[slot][:-1], target)
    adapted = type(jit)(None, CapturedJit(captured.ret, linear, list(names), new_info))
    print("INPUT_ONLY_ADAPTATION", "CPU", target, "PROGRAMS_UNCHANGED", len(programs),
          "BUFFERS_UNCHANGED", len(weights), flush=True)
    return adapted


def validate_host(artifact_path):
    import numpy as np
    from tinygrad import Device, Tensor
    from openpilot.selfdrive.modeld.helpers import load_oob
    from openpilot.selfdrive.modeld.compile_modeld import make_supercombo_input_queues
    assert Device["AMD"].arch == "gfx1102"
    print("LOADING_HOST_POLICY_INTERMEDIATE", flush=True)
    with artifact_path.open("rb") as handle:
        artifact = load_oob(handle)
    assert artifact["build_intermediate"]["deployable"] is False
    assert artifact["build_intermediate"]["source_input_device"] == "CPU"
    policy = artifact["run_policy"]
    adapted = retarget_warped_input(policy, "PYTHON")
    shapes = artifact["metadata"]["model"]["input_shapes"]

    def run(fn, source_device, seed):
        rng = np.random.default_rng(seed)
        queues, npy = make_supercombo_input_queues(shapes, artifact["frame_skip"], "AMD")
        for value in npy.values():
            value[:] = rng.standard_normal(value.shape).astype(value.dtype)
        frames = rng.integers(0, 256, size=(2, 6, 128, 256), dtype=np.uint8)
        warped = Tensor(frames, device=source_device).realize()
        outputs = fn(**{key: queues[key] for key in artifact["policy_input_keys"]}, warped=warped)
        result = [np.copy(t.numpy()) for t in outputs]
        assert all(np.isfinite(value).all() for value in result)
        return result

    for seed in (42, 43):
        expected = run(policy, "CPU", seed)
        actual = run(adapted, "PYTHON", seed)
        assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))
        print("HOST_SOURCE_DEVICE_ADAPTATION_BITWISE_MATCH", seed, flush=True)
    qcom_policy = retarget_warped_input(policy, "QCOM")
    assert qcom_policy.captured.expected_input_info[-1][-1] == "QCOM"
    assert "QCOM" not in Device._opened_devices
    print("QCOM_INPUT_CONTRACT_PREPARED_NOT_NATIVE_VALIDATED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--validate-host", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(DEV="USB+AMD:LLVM:gfx1102;CPU:LLVM", WARP_DEV="CPU",
                      JIT_BATCH_SIZE="0", GMMU="0", DEBUG="0", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(args.repo))
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    import openpilot.selfdrive.modeld.compile_modeld  # verified firmware cache handling
    validate_host(args.validate_host)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
