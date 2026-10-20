#!/usr/bin/env python3
"""Build a validated AMD policy intermediate; never installs a driving model.

Camera warps are deliberately omitted. The saved CPU-source input contract must
be adapted and validated with native QCOM warps before this can be deployed.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    if Path("/AGNOS").exists():
        raise RuntimeError("Heavy policy compilation must run on the PC/WSL, not comma hardware")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ.update(DEV="CPU:LLVM", WARP_DEV="CPU", FLOAT16="1",
                      JIT_BATCH_SIZE="0", GMMU="0", TC_OPT="2", DEBUG="1",
                      PYTHONDONTWRITEBYTECODE="1")
    for key in ("IMAGE", "NOLOCALS", "OPENPILOT_HACKS"):
        os.environ.pop(key, None)

    from bmrlnapv2_source import SOURCE_SHA, validate_source
    source = validate_source(args.stage)
    output = args.stage / "bmrlnapv2-gfx1102-HOST-POLICY-ONLY.pkl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    from tinygrad import Device, Tensor
    from tinygrad.helpers import DEV
    os.environ["DEV"] = "USB+AMD:LLVM:gfx1102;CPU:LLVM"
    DEV.value = os.environ["DEV"]
    import openpilot.selfdrive.modeld.compile_modeld as compiler
    gpu = Device["AMD"]
    assert gpu.arch == "gfx1102", gpu.arch
    assert (Tensor([1., 2., 3.], device="AMD") ** 2).sum().item() == 14.
    print("GPU_PREFLIGHT_OK gfx1102", flush=True)

    original_compile = compiler.compile_jit
    original_dump = compiler.dump_oob
    calls = 0

    def policy_only(*positional, **keywords):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_compile(*positional, **keywords)
        print("OMITTING_CAMERA_WARP: native QCOM build required", flush=True)
        return None

    def dump_intermediate(obj, handle):
        if isinstance(obj, dict) and "run_policy" in obj:
            assert calls == 3
            assert obj["metadata"]["model"]["model_checkpoint"] == (
                "594c0fe7-649a-4e06-5239-f3cc64a67aac/56320/"
                "2004ef3b-0461-b0f9-a1c6-23f444b6ef71/12864")
            obj = {key: value for key, value in obj.items() if not isinstance(key, tuple)}
            obj["build_intermediate"] = {
                "deployable": False,
                "reason": "Missing native QCOM warps and source-input device adaptation",
                "policy_arch": gpu.arch,
                "source_input_device": "CPU",
                "source_sha256": SOURCE_SHA,
            }
        return original_dump(obj, handle)

    compiler.compile_jit = policy_only
    compiler.dump_oob = dump_intermediate
    sys.argv = [str(compiler.__file__), "--model-type", "supercombo",
                "--model-size", "512x256", "--camera-resolutions",
                "1928x1208", "1344x760", "--frame-skip", "4",
                "--image-history-pipeline", "policy", "--behavior-version", "v16",
                "--out-of-band", "--supercombo-onnx", str(source),
                "--output", str(output)]
    result = compiler.main()
    print("HOST_POLICY_INTERMEDIATE_SAVED", output, flush=True)
    print("NOT_DEPLOYABLE: native QCOM assembly and validation still required", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
