#!/usr/bin/env python3
"""User-authorized temporary ForceOffroad transaction; always restores mode."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/data/openpilot")
    sys.path.insert(0, str(args.stage))
    import cereal.messaging as messaging
    from openpilot.common.params import Params
    from openpilot.starpilot.common.starpilot_variables import update_starpilot_toggles
    from build_bmrlnapv2_native_warps import require_offroad
    params = Params()
    sm = messaging.SubMaster(["deviceState", "pandaStates", "carState", "selfdriveState"])
    until = time.monotonic() + 8
    while time.monotonic() < until:
        sm.update(100)
        if all(sm.alive[s] and sm.valid[s] for s in ("deviceState", "pandaStates", "carState", "selfdriveState")):
            break
    else:
        raise RuntimeError("No fresh stationary/disengaged state; nothing changed")
    assert abs(float(sm["carState"].vEgo)) < 0.1, "Vehicle is moving"
    assert str(sm["carState"].gearShifter) == "park", "Vehicle is not in Park"
    assert not sm["selfdriveState"].enabled
    assert list(sm["pandaStates"]) and not any(p.controlsAllowed for p in sm["pandaStates"])
    print("PARK_ZERO_SPEED_DISENGAGED_VERIFIED", flush=True)
    keys = ("ForceOffroad", "ForceOnroad")
    original = {key: params.get(key) for key in keys}
    assert all(value is None or isinstance(value, bool) for value in original.values())
    recovery = args.stage / f"force-mode-recovery-{os.getpid()}.json"
    with recovery.open("x") as handle:
        json.dump({"pid": os.getpid(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   "original": original}, handle)
    active = None

    def abort(signum, _frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGINT, abort)
    try:
        params.put_bool("ForceOnroad", False)
        params.put_bool("ForceOffroad", True)
        update_starpilot_toggles()
        print("TEMPORARY_FORCE_OFFROAD_ENABLED_MANAGER_UNPAUSED", flush=True)
        until = time.monotonic() + 35
        while time.monotonic() < until:
            try:
                require_offroad()
                break
            except (AssertionError, RuntimeError):
                time.sleep(0.5)
        else:
            raise RuntimeError("Offroad/noOutput isolation did not become ready")
        for script in ("assemble_bmrlnapv2_native.py", "test_bmrlnapv2_modelstate.py"):
            require_offroad()
            print("RUNNING_STAGE", script, flush=True)
            active = subprocess.Popen([sys.executable, "-u", str(args.stage / script), "--stage", str(args.stage)])
            status = active.wait()
            active = None
            if status:
                raise RuntimeError(f"{script} exited with status {status}")
        print("ALL_STAGED_TESTS_COMPLETED_NO_INSTALL", flush=True)
    finally:
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait()
        for key, value in original.items():
            if value is None:
                params.remove(key)
            else:
                params.put_bool(key, value)
        update_starpilot_toggles()
        assert {key: params.get(key) for key in keys} == original, "Mode restoration failed"
        print("PREVIOUS_FORCE_MODES_RESTORED", original, flush=True)
        print("MANAGER_WAS_NEVER_PAUSED_SWAP_WAS_NOT_ENABLED", flush=True)


if __name__ == "__main__":
    main()
