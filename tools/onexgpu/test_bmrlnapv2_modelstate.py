#!/usr/bin/env python3
"""Offroad staged ModelState test with synthetic camera frames; no publishing."""
import argparse
import faulthandler
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    assert 1 <= args.seconds <= 60
    sys.path.insert(0, "/data/openpilot")
    sys.path.insert(0, str(args.stage))
    target = "QCOM;USB+AMD:LLVM:gfx1102;CPU:CLANG"
    os.environ.update(DEV=target, WARP_DEV="QCOM", PARALLEL="0", JIT_BATCH_SIZE="0",
                      GMMU="0", DEBUG="0", PYTHONDONTWRITEBYTECODE="1")
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    from build_bmrlnapv2_native_warps import require_offroad
    require_offroad()
    import numpy as np
    import cereal.messaging as messaging
    from openpilot.common.params import Params
    import openpilot.selfdrive.modeld.compile_modeld  # runtime firmware cache
    from openpilot.selfdrive.modeld.modeld import ModelState, _set_hcq_wait_timeout
    from openpilot.selfdrive.modeld.constants import ModelConstants
    from tinygrad.helpers import DEV
    DEV.value = target
    os.environ["DEV"] = target  # modeld import sets its production default
    _set_hcq_wait_timeout(30000)
    params = Params()
    keys = ("ActiveBigModel", "ActiveBigModelName", "ActiveSmallModelName", "ModelVersion", "DrivingModelVersion")
    before = {key: params.get(key) for key in keys}
    model_path = args.stage / "compiled" / "local-bmrlnapv2-onexgpu_driving_tinygrad.pkl"
    assert model_path.is_file()
    print("LOADING_STAGED_MODELSTATE", flush=True)
    model = ModelState(1928, 1208, True, model_id_override="local-bmrlnapv2-onexgpu",
                       write_model_version=False, model_version_override="v16",
                       model_path_override=model_path, force_external_gpu=True)
    assert model.uses_external_gpu and model.QUEUE_DEV == "AMD" and model.WARP_DEV == "QCOM"
    model.warmup()
    print("REAL_MODELSTATE_WARMUP_PASSED", flush=True)
    require_offroad()
    _set_hcq_wait_timeout(3000)
    rng = np.random.default_rng(42)
    frames = {key: rng.integers(0, 256, size=model.frame_buf_size, dtype=np.uint8)
              for key in model.vision_input_names}
    transforms = {key: np.eye(3, dtype=np.float32) for key in model.vision_input_names}
    inputs = {model.desire_key: np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)}
    for name, value in model.numpy_inputs.items():
        if name in (model.desire_key, model.prev_desired_curv_key):
            continue
        shape = value.shape[1:] if value.ndim > 1 and value.shape[0] == 1 else value.shape
        inputs[name] = np.zeros(shape, dtype=value.dtype)
    if "traffic_convention" in inputs:
        inputs["traffic_convention"].reshape(-1)[0] = 1
    sm = messaging.SubMaster(["deviceState", "pandaStates"])

    def guard():
        sm.update(0)
        assert all(sm.alive[s] and sm.valid[s] for s in ("deviceState", "pandaStates")), "Offroad state went stale"
        pandas = list(sm["pandaStates"])
        assert not sm["deviceState"].started and pandas
        forced = params.get_bool("ForceOffroad")
        assert all(not p.controlsAllowed and str(p.safetyModel) == "noOutput"
                   and (forced or (not p.ignitionLine and not p.ignitionCan))
                   for p in pandas), "Offroad state changed"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        sm.update(100)
        if all(sm.alive[s] and sm.valid[s] for s in ("deviceState", "pandaStates")):
            break
    durations = []
    start = time.monotonic()
    next_frame = start
    skipped = 0
    while time.monotonic() - start < args.seconds:
        guard()
        now = time.monotonic()
        if now < next_frame:
            time.sleep(next_frame - now)
        elif durations and now - next_frame >= 0.05:
            missed = int((now - next_frame) / 0.05)
            skipped += missed
            next_frame += missed * 0.05
        frame_start = time.perf_counter()
        result = model.run(frames, transforms, inputs, False)
        duration = (time.perf_counter() - frame_start) * 1000
        assert result is not None
        durations.append(duration)
        next_frame += 0.05
    elapsed = time.monotonic() - start
    guard()
    assert before == {key: params.get(key) for key in keys}, "Selected-model/version parameters changed"
    stats = {"test": "offroad_synthetic_frames_real_ModelState", "seconds": elapsed,
             "frames": len(durations), "synthetic_schedule_skips": skipped,
             "effective_hz": len(durations) / elapsed,
             "run_ms_p50": float(np.percentile(durations, 50)),
             "run_ms_p95": float(np.percentile(durations, 95)),
             "run_ms_p99": float(np.percentile(durations, 99)),
             "run_ms_max": max(durations),
             "runs_over_50ms": sum(value > 50 for value in durations),
             "selected_model_unchanged": True, "installed": False,
             "not_a_live_camera_or_driving_test": True}
    print("MODELSTATE_TEST_RESULT", json.dumps(stats, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
