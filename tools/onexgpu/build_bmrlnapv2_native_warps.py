#!/usr/bin/env python3
"""Offroad-only QCOM warp preparation; no AMD access or model installation."""
import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path


def require_offroad():
    import cereal.messaging as messaging
    from openpilot.common.params import Params
    sm = messaging.SubMaster(["deviceState", "pandaStates"])
    until = time.monotonic() + 5
    while time.monotonic() < until:
        sm.update(100)
        if not all(sm.alive[s] and sm.valid[s] for s in ("deviceState", "pandaStates")):
            continue
        pandas = list(sm["pandaStates"])
        assert not sm["deviceState"].started, "Comma is not offroad"
        forced = Params().get_bool("ForceOffroad")
        assert pandas and all(not p.controlsAllowed and str(p.safetyModel) == "noOutput"
                              and (forced or (not p.ignitionLine and not p.ignitionCan)) for p in pandas)
        break
    else:
        raise RuntimeError("No fresh offroad/noOutput state")
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        assert cmd[0] != b"selfdrive.modeld.modeld", "A driving-model process is running"
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    assert available > 512 * 1024, "Insufficient available RAM for isolated warp build"
    print("OFFROAD_NOOUTPUT_NO_MODELD", "MemAvailable_kB", available, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(DEV="QCOM;CPU:CLANG", WARP_DEV="QCOM", PARALLEL="0",
                      JIT_BATCH_SIZE="0", DEBUG="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("IMAGE", "NOLOCALS", "OPENPILOT_HACKS"):
        os.environ.pop(key, None)
    sys.path.insert(0, "/data/openpilot")
    sys.path.insert(0, str(args.stage))
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    require_offroad()
    output = args.stage / "bmrlnapv2-native-qcom-warps.pkl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    import numpy as np
    from tinygrad import Device, Tensor
    from tinygrad.engine.jit import TinyJit
    import openpilot.selfdrive.modeld.compile_modeld as compiler
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    from bmrlnapv2_source import EXPECTED, SOURCE_SHA

    def make_queues(_device):
        npy = {name: np.zeros((3, 3), dtype=np.float32) for name in ("tfm", "big_tfm")}
        return {name: Tensor(value, device="NPY").realize() for name, value in npy.items()}, npy

    bundle = {"kind": "bmrlnapv2-native-warps-only", "source_sha256": SOURCE_SHA,
              "model_checkpoint": EXPECTED["model_checkpoint"], "frame_skip": 4,
              "image_history_pipeline": "policy", "warp_input_keys": compiler.FAST_WARP_INPUTS,
              "image_shapes": EXPECTED["input_shapes"], "warps": {}}
    for cam_w, cam_h in ((1928, 1208), (1344, 760)):
        require_offroad()
        nv12 = compiler.NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
        print("BUILDING_NATIVE_QCOM_WARP", cam_w, cam_h, flush=True)
        jit = TinyJit(compiler.make_warp(nv12, 512, 256, 4, "policy"), prune=True)
        random_inputs = compiler.make_random_blob_images(
            ["frame", "big_frame"], nv12.size, device="QCOM")
        bundle["warps"][(cam_w, cam_h)] = compiler.compile_jit(
            jit, random_inputs, compiler.FAST_WARP_INPUTS, make_queues)
        assert "AMD" not in Device._opened_devices
    require_offroad()
    assert "AMD" not in Device._opened_devices
    with output.open("xb") as handle:
        compiler.dump_oob(bundle, handle)
    faulthandler.cancel_dump_traceback_later()
    print("NATIVE_QCOM_WARPS_SAVED", output, output.stat().st_size, flush=True)
    print("NO_SWAP_NO_SERVICE_CHANGES_NO_INSTALL", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
