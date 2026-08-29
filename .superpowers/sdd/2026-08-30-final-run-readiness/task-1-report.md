# Task 1 report: production monitored protocol and Mach contract

Date: 2026-08-30
Worktree: `C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830`
Branch: `codex/final-run-readiness`

## Scope and safety

Implemented only Task 1 from `docs/superpowers/plans/2026-08-30-final-run-readiness.md`.
No cloud resources were launched, no push or merge was performed, no subagents
were spawned, no generated build artifacts were edited, and the user's
`D:\CodeProjects\research-paper` checkout was not touched.

## TDD RED evidence

Focused tests were added/updated before the production implementation. The
first attempt exposed a test-collection error because a not-yet-created helper
was imported directly; the test was corrected to assert the missing behavior so
the RED run failed normally as required.

Command:

```text
python -m pytest -q tests/test_protocol_runner.py tests/test_runtime_config_wiring.py tests/test_config.py tests/test_constrained_recovery_review.py
```

Expected feature failures from the corrected RED run:

```text
FAILED tests/test_protocol_runner.py::TestProtocolRunner::test_gcp_128_protocol_uses_monitored_runner_with_explicit_production_contract
FAILED tests/test_runtime_config_wiring.py::test_monitored_training_config_does_not_override_sparse_runtime_schedule
FAILED tests/test_runtime_config_wiring.py::test_monitored_cfd_config_uses_explicit_backend_and_stream_block_size
FAILED tests/test_config.py::TestConfigSourceOfTruth::test_flow_fields_are_sourced_from_global_yaml
FAILED tests/test_constrained_recovery_review.py::test_resume_fingerprint_contains_live_training_behavior
FAILED tests/test_constrained_recovery_review.py::test_both_cfdconfig_classes_read_mach_number_from_config_yaml
6 failed, 51 passed, 3 warnings in 28.06s
```

The failures were the intended missing behaviors: GCP selected
`aircraft_diffusion_cfd.py`, monitored config ignored explicit BF16/cadence
values, the monitored CFD construction helper was absent, the direct batch
chunk was absent from the resume fingerprint, and both exact Mach expectations
still observed `0.3`.

## Implementation

- Added a `runner: monitored` protocol path in `CLI/run_protocol.py`.
- Wired the GCP protocol to `CLI/run_monitored_training.py` with explicit
  manifest, 12-epoch horizon, batch size, learning rate, latent/grid sizes,
  BF16 precision, D3Q27 solver, fused backend, coordinate/sparse/full/direct
  cadence, direct steps/directions/chunk, stream block size, checkpoint
  cadence, explicit direct-every-iteration policy, save directory, and bounded
  smoke stop.
- The enabled smoke block supplies a one-update atomic checkpoint cadence and
  a five-update bounded stop to the protocol command; the immutable training
  horizon remains 12 epochs.
- Added monitored-runner CLI arguments and construction for precision,
  coordinate sample count, sparse samples per full lattice, full-lattice
  interval, direct interval, direct steps/directions/chunk, stream block size,
  and direct-every-iteration policy.
- Applied the direct solver batch chunk to the existing runtime consumer and
  included it in the exact-resume objective fingerprint. The stream block size
  is carried in the explicit `LBMPhysicsConfig` nested in the CFD fingerprint.
- Changed the checked-in production CFD Mach value from `0.3` to `0.1` while
  preserving explicit constructor overrides and the no-silent-clamp tests.

## TDD GREEN evidence

Command:

```text
python -m pytest -q tests/test_protocol_runner.py tests/test_runtime_config_wiring.py tests/test_config.py tests/test_constrained_recovery_review.py
```

Output:

```text
57 passed, 3 warnings in 37.44s
```

The three warnings are the existing `pkg_resources` deprecation warnings from
the installed test environment; there were no test failures or collection
errors.

## Protocol dry-run evidence

Command:

```text
python CLI/run_protocol.py --config CLI/run_protocols/gcp_128_295m.yaml --dry-run
```

Output:

```text
$ 'C:\Program Files\Python312\python.exe' 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\CLI\run_monitored_training.py' --manifest 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\build\final_combined_corpus_20260824\combined_training_manifest.jsonl' --num-epochs 12 --batch-size 1 --learning-rate 2e-05 --latent-dim 512 --grid-size 128 --precision bfloat16 --solver D3Q27 --lbm-stream-bfl-backend fused_stream_bfl --coordinate-training-samples 65536 --sparse-samples-per-full 262144 --full-lattice-interval 64 --direct-solver-interval 32 --direct-solver-steps 5 --direct-solver-directions 8 --direct-solver-batch-chunk 4 --stream-block-size 512 --save-dir 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\checkpoints_128_295m' --checkpoint-every-updates 1 --no-require-direct-solver-every-iteration --stop-after-updates 5
```

The dry run exited successfully and did not execute the emitted training
command.

## Self-review

Command:

```text
git diff --check
```

Result: exit code `0`. Git emitted only its normal LF-to-CRLF working-copy
conversion warnings for the edited text files.

The staged scope is limited to the Task 1 implementation/tests and this
report. The supplied plan and the pre-existing `.superpowers/.../progress.md`
remain unstaged. Task 2 and Task 3 files/artifacts were not changed.

## Fix round 1/5 — review findings C1/C2/I1/I2/M1/M2

Review source:
`.superpowers/sdd/2026-08-30-final-run-readiness/task-1-review.md`.
The RED baseline was Task 1 commit `38a0935`.

### TDD RED evidence

Command:

```text
python -m pytest -q tests/test_protocol_runner.py tests/test_runtime_config_wiring.py tests/test_consistency_model.py::test_monitored_exact_resume_requires_every_expected_nested_identity_key tests/test_constrained_recovery_review.py::test_runner_main_restores_saved_threshold_and_resets_cadence tests/test_constrained_recovery_review.py::test_resume_fingerprint_contains_live_training_behavior
```

Output:

```text
7 failed, 6 passed, 3 warnings in 23.27s
```

The failures demonstrated the requested missing behaviors against `38a0935`:
the default protocol command still used smoke cadence, no explicit smoke mode
existed, grid spacing remained the fixed `0.01`, model runtime compile and
checkpointing settings had no monitored-runner wiring helper, monitored exact
resume accepted missing expected nested keys, the runner omitted the strict
compatibility schema, and checkpoint cadence was absent from resume identity.

### Minimal implementation

- C1: separated production and smoke command construction. Production is now
  the default, preserves the full 12-epoch run, uses checkpoint cadence 25,
  and has no bounded stop. `--mode smoke` must be selected explicitly and uses
  the configured one-update cadence and five-update stop.
- C2: added the monitored exact-resume schema and recursive validation that
  requires every immutable expected key, including nested keys, to be present
  and equal. Legacy run states without that schema retain the existing
  intersection-compatible behavior.
- I1: derive LBM grid spacing from physical length scale divided by resolved
  grid size, preserving the 128-resolution value `1 / 128`.
- I2: added explicit monitored-runner Boolean CLI flags and runtime application
  for model compilation and gradient checkpointing; protocol generation now
  emits both configured settings.
- M1: included `checkpoint_every_updates` in monitored resume identity.
- M2: corrected the stale test name and wording so the value is described as
  an explicit Mach `0.3` override rather than the documented default.

### TDD GREEN evidence

Initial focused command (same selection as RED):

```text
python -m pytest -q tests/test_protocol_runner.py tests/test_runtime_config_wiring.py tests/test_consistency_model.py::test_monitored_exact_resume_requires_every_expected_nested_identity_key tests/test_constrained_recovery_review.py::test_runner_main_restores_saved_threshold_and_resets_cadence tests/test_constrained_recovery_review.py::test_resume_fingerprint_contains_live_training_behavior
```

Output:

```text
13 passed, 3 warnings in 23.18s
```

After sharing the compatibility-schema constant and formatting the changes,
the affected regression selection was rerun:

```text
python -m pytest -q tests/test_protocol_runner.py tests/test_runtime_config_wiring.py tests/test_consistency_model.py::test_run_state_compatibility_configuration_uses_intersection_semantics tests/test_consistency_model.py::test_monitored_exact_resume_requires_every_expected_nested_identity_key tests/test_constrained_recovery_review.py::test_runner_main_restores_saved_threshold_and_resets_cadence tests/test_constrained_recovery_review.py::test_resume_fingerprint_contains_live_training_behavior
```

Output:

```text
14 passed, 3 warnings in 23.08s
```

Focused Task 1 suites:

```text
python -m pytest -q tests/test_protocol_runner.py tests/test_runtime_config_wiring.py tests/test_config.py tests/test_consistency_model.py tests/test_constrained_recovery_review.py
```

Output:

```text
101 passed, 3 warnings in 28.61s
```

Relevant exact-resume compatibility selection:

```text
python -m pytest -q tests/test_consistency_model.py tests/test_constrained_recovery_review.py -k "resume or run_state or compatibility"
```

Output:

```text
19 passed, 68 deselected, 3 warnings in 33.02s
```

The three warnings in each applicable run are the pre-existing installed
environment's `pkg_resources` deprecation warnings.

### Separate production and bounded-smoke command evidence

Production command:

```text
python CLI/run_protocol.py --config CLI/run_protocols/gcp_128_295m.yaml --mode production --dry-run
```

Output (single emitted command):

```text
$ 'C:\Program Files\Python312\python.exe' 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\CLI\run_monitored_training.py' --manifest 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\build\final_combined_corpus_20260824\combined_training_manifest.jsonl' --num-epochs 12 --batch-size 1 --learning-rate 2e-05 --latent-dim 512 --grid-size 128 --precision bfloat16 --solver D3Q27 --lbm-stream-bfl-backend fused_stream_bfl --coordinate-training-samples 65536 --sparse-samples-per-full 262144 --full-lattice-interval 64 --direct-solver-interval 32 --direct-solver-steps 5 --direct-solver-directions 8 --direct-solver-batch-chunk 4 --stream-block-size 512 --save-dir 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\checkpoints_128_295m' --checkpoint-every-updates 25 --no-require-direct-solver-every-iteration --enable-compile --enable-gradient-checkpointing
```

Bounded smoke command:

```text
python CLI/run_protocol.py --config CLI/run_protocols/gcp_128_295m.yaml --mode smoke --dry-run
```

Output (single emitted command):

```text
$ 'C:\Program Files\Python312\python.exe' 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\CLI\run_monitored_training.py' --manifest 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\build\final_combined_corpus_20260824\combined_training_manifest.jsonl' --num-epochs 12 --batch-size 1 --learning-rate 2e-05 --latent-dim 512 --grid-size 128 --precision bfloat16 --solver D3Q27 --lbm-stream-bfl-backend fused_stream_bfl --coordinate-training-samples 65536 --sparse-samples-per-full 262144 --full-lattice-interval 64 --direct-solver-interval 32 --direct-solver-steps 5 --direct-solver-directions 8 --direct-solver-batch-chunk 4 --stream-block-size 512 --save-dir 'C:\Users\Darsh Gupta\AppData\Local\Temp\research-paper-final-review-20260830\checkpoints_128_295m' --checkpoint-every-updates 1 --no-require-direct-solver-every-iteration --enable-compile --enable-gradient-checkpointing --stop-after-updates 5
```

Both dry runs exited successfully without launching training or cloud
resources. The production command has no bounded-stop argument; the smoke
command remains explicitly bounded.

### Fix-round self-review

- Reviewed every production and test diff for C1, C2, I1, I2, M1, and M2.
- Confirmed the strict schema affects monitored exact-resume state while the
  compatibility test preserves legacy intersection semantics.
- Confirmed the production and smoke commands remain separate and include all
  configured immutable runtime fields.
- `git diff --check` exited `0`; Git printed only LF-to-CRLF working-copy
  conversion warnings.
- Final pre-commit rerun: the focused Task 1 suites passed `101 passed,
  3 warnings in 28.37s`; the exact-resume selection passed `19 passed,
  68 deselected, 3 warnings in 19.00s`; both production and smoke dry runs
  exited `0` with the command shapes recorded above.
- No cloud resource, training run, push, merge, generated build artifact, or
  user dirty checkout was touched.
