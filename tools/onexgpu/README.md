# ONEXGPU gfx1102 build and parked-test helpers

These preserve the StarPilot bench workflow validated against StarPilot commit
`725ea67ee62da1924e3e4b2f3a15a8e207767a17`. They are not sunnypilot
installation scripts, and must not be assumed compatible with another runner.

## Scope and precautions

- Original ONEXGPU / RX 7600M XT: `gfx1102`; published `gfx1200` binaries are not interchangeable.
- Heavy ONNX/policy compilation runs on a PC/WSL with the GPU attached.
- Native QCOM warp assembly and tests require parked/offroad, fresh Panda
  `noOutput`, controls disallowed, and no competing driving-model process.
- Do not enable disk swap for native model compilation: an earlier attempt
  triggered an AGNOS `pfk_bio_get_inode` kernel crash during swap writeback.
  That failed launcher is deliberately omitted.
- Do not pause the manager. Temporary ForceOffroad wrappers keep it running and
  restore the original mode values. Interrupted tests still require checking the
  saved recovery state and current modes before returning the vehicle to use.
- Never run Python with `-O`: these tools use assertions for important checks.
- Parked performance tests do not establish road safety or suitability. Do not
  engage assistance during these tests.
- Only load trusted pickle/model artifacts. No model binaries, route recordings,
  keys, private settings, or recovery-state files are included here.

## Workflow

Keep these helper modules together in a private staging directory. Each entry
point accepts `--help`. Use the matching StarPilot source and its native/WSL
Python environment; these depend on its internal tinygrad and model APIs.

1. `bmrlnapv2_source.py` validates the exact ONNX hash, checkpoint and shapes.
2. `build_bmrlnapv2_host_policy.py` builds a **non-deployable** host intermediate.
3. `adapt_bmrlnapv2_policy_input.py` verifies input-only retargeting using two
   seeds and bitwise-matching outputs; GPU programs and weights are unchanged.
4. `build_bmrlnapv2_native_warps.py` builds only QCOM warps on the comma.
5. `assemble_bmrlnapv2_native.py` checks both component hashes, assembles them,
   and verifies finite outputs for both camera resolutions before saving.
6. `test_bmrlnapv2_modelstate.py` exercises the real ModelState offroad without publishing.
7. `run_bmrlnapv2_forced_offroad.py` wraps assembly/testing in a parked,
   temporary ForceOffroad transaction.
8. `install_and_test_bmrlnapv2.py` is a **mutating, one-time StarPilot-only**
   registration/selection tool and parked live-camera test. It preserves the
   small fallback and restores the prior selection if validation fails.
9. `run_pipeline.py` is the earlier **host-only v9 replay** benchmark. Its local
   replay/artifact inputs must already exist; it is not the BMRLNAP V2 live test.

The common metadata helpers were factored out of the disabled native launcher;
the host-only entry points additionally refuse to run on AGNOS.

## Exact artifacts

Source ONNX: [BMRLNAP V2 release](https://github.com/happymaj11r/openpilot-models/releases/download/big-remove-avgpool-6afb28d5/big_driving_supercombo.onnx)

| Artifact | SHA-256 |
| --- | --- |
| Source ONNX | `6afb28d58bfa7d7639832db8a61ff9209ef704ce1a9de2cdc20dd9004ec42bec` |
| Host policy intermediate | `4786f8073c9b9b47fadda98724b1181f3e0cbc814770cb58733cf27e48957b18` |
| Native QCOM warps | `18df514d755e92b0bf4138fcf4f26e099cd8cc34b238e8ad8ff8292397bc0222` |
| Assembled ONEXGPU model | `ad893dea7371170ea0cb6cb9326622641bf6c113a203cc91cbade376a437337d` |

The original parked StarPilot live-camera check published 1,200 model messages
in 60.002 seconds (~20 Hz), with no missing frame IDs and a 46.405 ms median
execution time. This records that test only; it is not a sunnypilot validation.
