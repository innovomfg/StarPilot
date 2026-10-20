#!/usr/bin/env python3
"""Assemble and test staged QCOM+gfx1102 components. Never installs/selects."""
import argparse
import faulthandler
import hashlib
import os
import sys
import time
from pathlib import Path

HOST_SHA = "4786f8073c9b9b47fadda98724b1181f3e0cbc814770cb58733cf27e48957b18"
WARP_SHA = "18df514d755e92b0bf4138fcf4f26e099cd8cc34b238e8ad8ff8292397bc0222"


def checked_hash(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == expected, f"Checksum mismatch: {path}"
    print("CHECKSUM_VERIFIED", path.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/data/openpilot")
    sys.path.insert(0, str(args.stage))
    os.environ.update(DEV="QCOM;USB+AMD:LLVM:gfx1102;CPU:CLANG", WARP_DEV="QCOM",
                      PARALLEL="0", JIT_BATCH_SIZE="0", GMMU="0", DEBUG="0",
                      PYTHONDONTWRITEBYTECODE="1")
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    from build_bmrlnapv2_native_warps import require_offroad
    from adapt_bmrlnapv2_policy_input import retarget_warped_input
    from bmrlnapv2_source import EXPECTED, SOURCE_SHA, normalize
    require_offroad()
    assert len(Path("/proc/swaps").read_text().splitlines()) == 1, "Swap must remain disabled"
    host = args.stage / "bmrlnapv2-gfx1102-HOST-POLICY-ONLY.pkl"
    warps_path = args.stage / "bmrlnapv2-native-qcom-warps.pkl"
    output = args.stage / "compiled" / "local-bmrlnapv2-onexgpu_driving_tinygrad.pkl"
    assert not output.exists(), "Refusing to overwrite an assembled model"
    checked_hash(host, HOST_SHA)
    checked_hash(warps_path, WARP_SHA)

    import numpy as np
    from tinygrad import Device, Tensor
    import openpilot.selfdrive.modeld.compile_modeld as compiler
    from openpilot.selfdrive.modeld.helpers import load_oob, dump_oob
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    require_offroad()
    gpu = Device["AMD"]
    assert gpu.arch == "gfx1102"
    assert (Tensor([1., 2., 3.], device="AMD") ** 2).sum().item() == 14.
    print("NATIVE_AMD_PREFLIGHT_OK", gpu.arch, flush=True)
    with host.open("rb") as handle:
        artifact = load_oob(handle)
    assert artifact["build_intermediate"] == {
        "deployable": False,
        "reason": "Missing native QCOM warps and source-input device adaptation",
        "policy_arch": "gfx1102", "source_input_device": "CPU", "source_sha256": SOURCE_SHA}
    assert normalize(artifact["metadata"]["model"]) == EXPECTED
    assert artifact["frame_skip"] == 4 and artifact["image_history_pipeline"] == "policy"
    with warps_path.open("rb") as handle:
        native = load_oob(handle)
    assert native["source_sha256"] == SOURCE_SHA
    assert native["model_checkpoint"] == EXPECTED["model_checkpoint"]
    assert native["image_shapes"] == EXPECTED["input_shapes"]
    assert native["frame_skip"] == artifact["frame_skip"]
    assert native["image_history_pipeline"] == artifact["image_history_pipeline"]
    assert tuple(native["warp_input_keys"]) == tuple(artifact["warp_input_keys"])
    artifact["run_policy"] = retarget_warped_input(artifact["run_policy"], "QCOM")
    artifact.update(native["warps"])

    for cam_w, cam_h in ((1928, 1208), (1344, 760)):
        require_offroad()
        nv12 = compiler.NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
        make_frames = compiler.make_random_blob_images(["frame", "big_frame"], nv12.size, "QCOM")
        results = []
        for seed in (42, 43):
            np.random.seed(seed)
            queues, npy = compiler.make_supercombo_input_queues(EXPECTED["input_shapes"], 4, "AMD")
            npy["tfm"][:] = np.eye(3, dtype=np.float32)
            npy["big_tfm"][:] = np.eye(3, dtype=np.float32)
            frames = make_frames()
            start = time.perf_counter()
            warped = artifact[(cam_w, cam_h)](tfm=queues["tfm"], big_tfm=queues["big_tfm"], **frames)
            assert warped.device == "QCOM" and tuple(warped.shape) == (2, 6, 128, 256)
            tensors = artifact["run_policy"](
                **{key: queues[key] for key in artifact["policy_input_keys"]}, warped=warped)
            values = [np.copy(value.numpy()) for value in tensors]
            elapsed = (time.perf_counter() - start) * 1000
            assert len(values) == 1 and values[0].shape == (1, 18452)
            assert np.isfinite(values[0]).all()
            results.append(values[0])
            print("NATIVE_COMPONENT_PIPELINE_PASS", cam_w, cam_h, "seed", seed,
                  "ms", round(elapsed, 2), flush=True)
        assert not np.array_equal(*results), "Pipeline output did not respond to changed frames"

    require_offroad()
    artifact.pop("build_intermediate")
    artifact["build_provenance"] = {
        "source_sha256": SOURCE_SHA, "host_policy_sha256": HOST_SHA,
        "native_warps_sha256": WARP_SHA, "policy_arch": "gfx1102",
        "input_only_adaptation": "CPU-to-QCOM", "native_component_pipeline_passed": True,
        "modelstate_warmup_pending": True}
    with output.open("xb") as handle:
        dump_oob(artifact, handle)
    faulthandler.cancel_dump_traceback_later()
    print("STAGED_NATIVE_MODEL_SAVED", output, output.stat().st_size, flush=True)
    print("NOT_INSTALLED_NOT_SELECTED_MODELSTATE_TEST_STILL_REQUIRED", flush=True)


if __name__ == "__main__":
    main()
