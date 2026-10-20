#!/usr/bin/env python3
"""Install a separate local model and validate normal modeld while parked."""
import argparse
import hashlib
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path

KEY = "local-bmrlnapv2-onexgpu"
NAME = "BMRLNAP V2 ONEXGPU (gfx1102)"
CATALOG = {"AvailableModels": KEY, "AvailableModelNames": NAME,
           "AvailableModelSeries": "Local", "AvailableModelArtifactFormats": None,
           "ModelReleasedDates": "2026-10-20", "ModelVersions": "v16"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/data/openpilot")
    sys.path.insert(0, str(args.stage))
    import numpy as np
    import cereal.messaging as messaging
    from openpilot.common.params import Params
    from openpilot.starpilot.common.starpilot_variables import update_starpilot_toggles
    from openpilot.starpilot.common.model_versions import UNIFIED_ARTIFACT_FORMAT
    from openpilot.starpilot.assets.model_manager import set_model_profile, get_model_profile, model_uses_external_gpu
    from build_bmrlnapv2_native_warps import require_offroad
    params = Params()
    services = ["deviceState", "pandaStates", "carState", "selfdriveState", "modelV2", "roadCameraState", "chestnutState"]
    sm = messaging.SubMaster(services, poll="modelV2")

    def parked(strict=True):
        assert sm.alive["pandaStates"] and sm.valid["pandaStates"], "Panda state stale"
        pandas = list(sm["pandaStates"])
        assert pandas and not any(p.controlsAllowed for p in pandas), "Controls are allowed"
        for service in ("carState", "selfdriveState"):
            if strict:
                assert sm.alive[service] and sm.valid[service], f"{service} stale"
        if sm.alive["carState"] and sm.valid["carState"]:
            assert abs(float(sm["carState"].vEgo)) < 0.1 and str(sm["carState"].gearShifter) == "park", "Vehicle left Park"
        if sm.alive["selfdriveState"] and sm.valid["selfdriveState"]:
            assert not sm["selfdriveState"].enabled, "Assistance engaged"

    until = time.monotonic() + 8
    while time.monotonic() < until:
        sm.update(100)
        if all(sm.alive[s] and sm.valid[s] for s in ("pandaStates", "carState", "selfdriveState")):
            break
    parked()
    print("PARK_ZERO_SPEED_DISENGAGED_VERIFIED", flush=True)
    mode_keys = ("ForceOffroad", "ForceOnroad")
    big_keys = ("ActiveBigModel", "ActiveBigModelName", "ActiveBigModelVersion")
    small_keys = ("ActiveSmallModel", "ActiveSmallModelName", "ActiveSmallModelVersion")
    modes = {key: params.get(key) for key in mode_keys}
    original_big = {key: params.get(key) for key in big_keys}
    original_small = {key: params.get(key) for key in small_keys}
    assert not modes["ForceOffroad"], "Live-camera test requires the original normal mode"
    columns = {key: (params.get(key) or "").split(",") for key in CATALOG}
    assert len({len(values) for values in columns.values()}) == 1, "Catalog columns are misaligned"
    assert KEY not in columns["AvailableModels"], "Local model already registered; inspect before rerunning"
    models = Path("/data/models")
    source = args.stage / "compiled" / f"{KEY}_driving_tinygrad.pkl"
    dest = models / source.name
    sidecar = models / f"{KEY}.json"
    assert source.stat().st_size == 1780715121
    assert not dest.exists() and not sidecar.exists(), "Refusing to overwrite an existing local model"
    with source.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little", signed=True)
        opcodes = handle.read(size)
    assert b"gfx1102" in opcodes and b"gfx1200" not in opcodes and b"build_provenance" in opcodes
    backup = args.stage / f"installation-backup-{os.getpid()}"
    backup.mkdir()
    cache_paths = [models / name for name in (".model_artifacts.json", ".model_versions.json")]
    cache_data = {}
    for path in cache_paths:
        data = json.loads(path.read_text()) if path.exists() else {}
        assert isinstance(data, dict)
        cache_data[path.name] = data
        if path.exists():
            shutil.copy2(path, backup / path.name)
    write_json(backup / "params.json", {"modes": modes, "big": original_big, "small": original_small,
                                        "catalog": {key: params.get(key) for key in CATALOG}})
    profile_changed = False
    completed = False

    def abort(signum, _frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGINT, abort)
    signal.signal(signal.SIGTERM, abort)

    def restore_modes():
        for key, value in modes.items():
            if value is None:
                params.remove(key)
            else:
                params.put_bool(key, value)
        update_starpilot_toggles()

    def isolate():
        params.put_bool("ForceOnroad", False)
        params.put_bool("ForceOffroad", True)
        update_starpilot_toggles()
        until = time.monotonic() + 40
        while time.monotonic() < until:
            try:
                require_offroad()
                return
            except (AssertionError, RuntimeError):
                time.sleep(0.5)
        raise RuntimeError("Offroad isolation did not become ready")

    try:
        isolate()
        source_sha = digest(source)
        temporary = models / f".{source.name}.installing-{os.getpid()}"
        with source.open("rb") as inp, temporary.open("xb") as out:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        assert digest(temporary) == source_sha
        assert not dest.exists()
        temporary.rename(dest)
        info = {"name": NAME, "series": "Local", "version": "v16", "released": "2026-10-20",
                "uses_external_gpu": True, "model_size": "chestnut", "model_lab_eligible": False,
                "artifact_sha256": source_sha, "policy_arch": "gfx1102"}
        write_json(sidecar, info)
        metadata = json.loads((models / ".model_artifacts.json").read_text()) if (models / ".model_artifacts.json").exists() else {}
        assert isinstance(metadata, dict)
        metadata[KEY] = {"artifact_format": UNIFIED_ARTIFACT_FORMAT, "artifact_filename": dest.name,
                         "artifact_size": dest.stat().st_size, "artifact_sha256": source_sha,
                         "artifact_chunk_count": 0, "artifact_url": "", "uses_external_gpu": True,
                         "model_size": "chestnut", "model_size_declared": True,
                         "model_lab_eligible": False, "accelerator_artifacts": {}}
        write_json(models / ".model_artifacts.json", metadata)
        versions = json.loads((models / ".model_versions.json").read_text()) if (models / ".model_versions.json").exists() else {}
        assert isinstance(versions, dict)
        versions[KEY] = "v16"
        write_json(models / ".model_versions.json", versions)
        CATALOG["AvailableModelArtifactFormats"] = UNIFIED_ARTIFACT_FORMAT
        for key, value in CATALOG.items():
            params.put(key, ",".join([*columns[key], value]))
        assert model_uses_external_gpu(KEY)
        set_model_profile(params, "big", KEY, NAME, "v16")
        profile_changed = True
        assert get_model_profile(params, "big") == (KEY, NAME, "v16")
        assert {key: params.get(key) for key in small_keys} == original_small
        update_starpilot_toggles()
        print("LOCAL_MODEL_INSTALLED_AND_SELECTED", NAME, "SHA256", source_sha, flush=True)
        restore_modes()
        print("NORMAL_MODE_RESTORED_WAITING_FOR_LIVE_GPU_MODEL", flush=True)
        ready_start = time.monotonic()
        last_status = 0
        while time.monotonic() - ready_start < 180:
            sm.update(100)
            parked(strict=False)
            active = params.get_bool("UsbGpuActive")
            runtime = params.get("DrivingModel")
            if active and runtime == KEY and sm.alive["modelV2"] and sm.valid["modelV2"]:
                parked()
                break
            if time.monotonic() - last_status > 10:
                print("LIVE_LOAD_STATUS", runtime, "active", active, "loading", params.get_bool("UsbGpuLoading"), flush=True)
                last_status = time.monotonic()
        else:
            raise RuntimeError("Installed GPU model did not become the live active model")
        print("LIVE_GPU_MODEL_ACTIVE", params.get("DrivingModelVersion"), flush=True)
        warm_until = time.monotonic() + 10
        while time.monotonic() < warm_until:
            sm.update(100)
            parked()
        samples = []
        temperatures = []
        first_camera = None
        last_camera = None
        measurement_start = time.monotonic()
        last_status = measurement_start
        while time.monotonic() - measurement_start < 60:
            sm.update(100)
            parked()
            now = time.monotonic()
            if sm.updated["roadCameraState"]:
                camera = int(sm["roadCameraState"].frameId)
                first_camera = camera if first_camera is None else first_camera
                last_camera = camera
            if sm.updated["modelV2"]:
                assert sm.valid["modelV2"]
                model = sm["modelV2"]
                duration = float(model.modelExecutionTime) * 1000
                assert np.isfinite(duration)
                samples.append((int(model.frameId), now, duration, float(model.frameDropPerc)))
            if sm.updated["chestnutState"] and sm.valid["chestnutState"]:
                temperatures.append(float(sm["chestnutState"].tempC))
            if now - last_status > 10:
                assert params.get_bool("UsbGpuActive") and params.get("DrivingModel") == KEY, "GPU runtime fell back"
                print("LIVE_CAMERA_PROGRESS", round(now - measurement_start, 1), "seconds", len(samples), "published_models", flush=True)
                last_status = now
        elapsed = time.monotonic() - measurement_start
        assert samples and params.get_bool("UsbGpuActive") and params.get("DrivingModel") == KEY
        durations = [s[2] for s in samples]
        drops = [s[3] for s in samples]
        span = samples[-1][0] - samples[0][0] + 1
        missing = max(0, span - len(samples))
        result = {"model": KEY, "test": "parked_live_cameras_normal_modeld", "seconds": elapsed,
                  "published_models": len(samples), "published_hz": len(samples) / elapsed,
                  "frame_id_span": span, "missing_model_frame_ids": missing,
                  "missing_model_fraction": missing / span, "camera_frame_span": (last_camera - first_camera + 1) if first_camera is not None else None,
                  "execution_ms_p50": float(np.percentile(durations, 50)),
                  "execution_ms_p95": float(np.percentile(durations, 95)),
                  "execution_ms_p99": float(np.percentile(durations, 99)),
                  "execution_ms_max": max(durations), "execution_over_50ms": sum(v > 50 for v in durations),
                  "reported_drop_percent_p95": float(np.percentile(drops, 95)),
                  "reported_drop_percent_max": max(drops),
                  "gpu_temp_c_max": max(temperatures) if temperatures else None,
                  "gpu_active": True, "assistance_remained_disengaged": True}
        result["parked_performance_check_passed"] = result["published_hz"] >= 19.5 and missing / span <= 0.02 and result["reported_drop_percent_p95"] <= 2.0
        write_json(args.stage / "live-camera-test-result.json", result)
        print("LIVE_CAMERA_TEST_RESULT", json.dumps(result, sort_keys=True), flush=True)
        assert {key: params.get(key) for key in small_keys} == original_small
        if not result["parked_performance_check_passed"]:
            raise RuntimeError("Live-camera throughput/drop check failed; restoring prior selection")
        completed = True
        print("LOCAL_MODEL_INSTALL_AND_PARKED_TEST_PASSED", flush=True)
    finally:
        try:
            if profile_changed and not completed:
                sm.update(100)
                parked(strict=False)
                isolate()
                for key, value in original_big.items():
                    if value is None:
                        params.remove(key)
                    else:
                        params.put(key, value)
                update_starpilot_toggles()
                print("PRIOR_BIG_PROFILE_RESTORED_LOCAL_MODEL_RETAINED", original_big, flush=True)
        finally:
            restore_modes()
            assert {key: params.get(key) for key in mode_keys} == modes
            print("PREVIOUS_FORCE_MODES_RESTORED", modes, flush=True)
            print("NO_MANAGER_PAUSE_NO_SWAP", flush=True)


if __name__ == "__main__":
    main()
