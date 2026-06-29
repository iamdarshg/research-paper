# Expanded GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a fresh `96^3`, 7-8M-parameter aircraft training run over at least 600 unique grounded geometries, with exact GPU-accelerated boundary preparation and every computed training term incorporated into the optimization loss.

**Architecture:** Keep the existing sequential base/plus/minus SPSA objective and FP32 D3Q27 physics. Add a compact deduplicated host geometry store, a 96-dimensional latent model with a Fourier residual coordinate decoder, an exact CuPy/SciPy SDF backend boundary, and a parity-gated Triton BFL path. Training uses only loss-bearing computations; report-only solver analysis remains in explicit evaluation commands.

**Tech Stack:** Python 3.12, PyTorch, CuPy CUDA 12, Triton, SciPy reference EDT, NumPy, trimesh, pytest, Click.

---

### Task 1: Compact Unique Geometry Store

**Files:**
- Create: `CLI/geometry_store.py`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Test: `tests/test_geometry_store.py`

- [ ] **Step 1: Write failing compact-store tests**

```python
def test_store_deduplicates_content_and_keeps_uint8():
    geometry = torch.zeros((8, 8, 8))
    geometry[2:6, 3:5, 1:7] = 1
    store = CompactGeometryStore()
    first = store.add("a", geometry, content_hash="same")
    second = store.add("b", geometry.clone(), content_hash="same")
    assert first == second
    assert store.unique_count == 1
    assert store.get(first).dtype == torch.uint8


def test_dataset_records_reference_shared_geometry():
    store = CompactGeometryStore()
    index = store.add("a", torch.ones((4, 4, 4)), content_hash="same")
    assert store.materialize(index).data_ptr() == store.materialize(index).data_ptr()
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_geometry_store.py -q`

Expected: import failure because `CLI.geometry_store` does not exist.

- [ ] **Step 3: Implement the compact store**

```python
class CompactGeometryStore:
    def __init__(self):
        self._geometries: list[torch.Tensor] = []
        self._hash_to_index: dict[str, int] = {}

    def add(self, source_id: str, geometry: torch.Tensor, *, content_hash: str) -> int:
        existing = self._hash_to_index.get(content_hash)
        if existing is not None:
            return existing
        compact = (geometry.detach().cpu() > 0.5).to(torch.uint8).contiguous()
        index = len(self._geometries)
        self._geometries.append(compact)
        self._hash_to_index[content_hash] = index
        return index

    def materialize(self, index: int) -> torch.Tensor:
        return self._geometries[index]

    @property
    def unique_count(self) -> int:
        return len(self._geometries)
```

Update `AircraftDesignDataset` so manifest records store geometry indices and
`__getitem__` resolves the shared uint8 tensor. Do not convert the full corpus to
float32 on load.

- [ ] **Step 4: Add pinned-transfer tests**

Test that `aircraft_collate_fn` preserves compact uint8 geometry and that the
training transfer converts it once on the destination device with
`non_blocking=True`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_geometry_store.py tests/test_cli.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add CLI/geometry_store.py CLI/aircraft_diffusion_cfd.py tests/test_geometry_store.py
git commit -m "perf: deduplicate compact host geometries"
```

### Task 2: Unique-Geometry Corpus Expansion

**Files:**
- Modify: `CLI/build_hiliftaeroml_voxel_manifest.py`
- Modify: `CLI/validate_manifest.py`
- Test: `tests/test_hiliftaeroml_voxel_manifest.py`
- Test: `tests/test_manifest_validation.py`

- [ ] **Step 1: Write failing unique-selection tests**

```python
def test_selection_uses_one_surface_per_new_geometry_variant():
    selected = select_unique_hilift_variants(
        catalog_records=records_with_repeated_aoa(),
        existing_variant_ids={"LHC001"},
        target_unique_geometries=3,
    )
    assert [row["geometry_variant_id"] for row in selected] == ["LHC002", "LHC003"]
    assert len({row["geometry_variant_id"] for row in selected}) == len(selected)


def test_manifest_gate_requires_600_unique_geometries(tmp_path):
    report = validate_manifest_records(repeated_records(600, unique=599))
    assert report["status"] == "fail"
    assert report["unique_geometry_count"] == 599
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_hiliftaeroml_voxel_manifest.py tests/test_manifest_validation.py -q`

Expected: missing unique-selector/gate behavior.

- [ ] **Step 3: Implement canonical variant selection**

Select one canonical AoA record per `geometry_variant_id`, skip existing variant
IDs and content hashes, and stop only when the requested unique target is met.
Add CLI option:

```python
parser.add_argument("--target-unique-geometries", type=_positive_int, default=600)
```

- [ ] **Step 4: Implement fail-closed validation**

The validation report must include:

```python
{
    "record_count": len(records),
    "unique_geometry_count": len(unique_hashes),
    "duplicate_geometry_record_count": len(records) - len(unique_hashes),
    "unique_geometry_target": 600,
    "unique_geometry_target_met": len(unique_hashes) >= 600,
}
```

Final-run validation returns failure when the target is not met.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_hiliftaeroml_voxel_manifest.py tests/test_manifest_validation.py -q`

Expected: all pass.

- [ ] **Step 6: Build the expanded corpus**

Run the builder against the exact-CAD catalog with a new dated output root and
`--target-unique-geometries 600`. Preserve source URLs, license fields, source
hashes, variant IDs, family IDs, and grouped splits. Reject duplicate content,
component-only CAD, invalid meshes, and failed voxelizations.

- [ ] **Step 7: Validate the corpus**

Run `CLI/validate_manifest.py` and the aircraft validity screen. Record exact
accepted, rejected, duplicate, source, family, and split counts.

- [ ] **Step 8: Commit code and corpus documentation**

```bash
git add CLI/build_hiliftaeroml_voxel_manifest.py CLI/validate_manifest.py tests/test_hiliftaeroml_voxel_manifest.py tests/test_manifest_validation.py docs/dataset
git commit -m "feat: require 600 unique grounded geometries"
```

Do not commit downloaded binary CAD or voxel build artifacts.

### Task 3: 7-8M Parameter Fourier Geometry Model

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_consistency_model.py`

- [ ] **Step 1: Write failing model-configuration tests**

```python
def test_expanded_model_parameter_budget():
    config = ModelConfig.expanded_g96()
    models = build_training_models(config, DiffusionConfig(), TrainingConfig())
    count = count_unique_trainable_parameters(models)
    assert 7_000_000 <= count <= 8_000_000


def test_fourier_decoder_preserves_grid_shape():
    converter = LatentTo3DConverter(
        latent_dim=96,
        grid_resolution=16,
        coordinate_decoder_threshold=1,
        coordinate_decoder_width=512,
        coordinate_decoder_depth=4,
        coordinate_fourier_bands=6,
    )
    assert converter(torch.zeros((2, 96))).shape == (2, 16, 16, 16)
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_cli.py tests/test_consistency_model.py -q`

Expected: missing expanded configuration and decoder arguments.

- [ ] **Step 3: Add serialized model fields**

Add `coordinate_decoder_width`, `coordinate_decoder_depth`,
`coordinate_fourier_bands`, and `coordinate_chunk_size` to `ModelConfig`.
Provide an `expanded_g96()` constructor with latent width 96 and channels
`[64, 96, 128]` / `[128, 96, 64]`.

- [ ] **Step 4: Implement Fourier residual decoding**

Encode each coordinate as raw xyz plus
`sin(2^k*pi*x), cos(2^k*pi*x)` for six bands. Build a 512-wide input layer,
four residual SiLU hidden layers, and a scalar output. Apply activation
checkpointing during gradient-bearing full-grid decoding.

- [ ] **Step 5: Preserve old checkpoint compatibility**

When checkpoint payloads omit the new fields, instantiate the previous
raw-coordinate 256-wide two-layer decoder exactly. Add a round-trip test for
both legacy and expanded configurations.

- [ ] **Step 6: Run focused tests and count parameters**

Run: `python -m pytest tests/test_cli.py tests/test_consistency_model.py -q`

Expected: all pass and expanded count is within 7-8M.

- [ ] **Step 7: Commit**

```bash
git add CLI/aircraft_diffusion_cfd.py tests/test_cli.py tests/test_consistency_model.py
git commit -m "feat: add expanded Fourier geometry model"
```

### Task 4: Exact GPU SDF Boundary Backend

**Files:**
- Modify: `CLI/sdf_utils.py`
- Modify: `requirements.txt`
- Test: `tests/test_sdf_utils.py`

- [ ] **Step 1: Write failing backend/parity tests**

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_exact_sdf_matches_scipy_reference():
    geometry = deterministic_aircraft_mask(32).cuda()
    reference = compute_sdf(geometry, backend="scipy_reference")
    actual = compute_sdf(geometry, backend="gpu_exact")
    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=1e-5, atol=1e-5)


def test_gpu_backend_fails_closed_when_unavailable(monkeypatch):
    monkeypatch.setattr(sdf_utils, "_cupy_available", lambda: False)
    with pytest.raises(RuntimeError, match="gpu_exact"):
        compute_sdf(torch.zeros((8, 8, 8)), backend="gpu_exact")
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_sdf_utils.py -q`

Expected: `backend` argument is unsupported.

- [ ] **Step 3: Add CuPy exact EDT with DLPack**

Use `cupyx.scipy.ndimage.distance_transform_edt` with float32 distances.
Convert PyTorch/CuPy arrays through DLPack without CPU materialization. Compute
inside and outside transforms sequentially to bound workspace memory. Keep
`scipy_reference` unchanged.

- [ ] **Step 4: Add startup probe and explicit fallback**

`backend="auto"` may select `gpu_exact` only after a deterministic startup
parity probe. Explicit `gpu_exact` failure raises; it never silently switches
to an approximate method.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_sdf_utils.py -q`

Expected: CPU tests pass; CUDA parity passes on the target GPU.

- [ ] **Step 6: Commit**

```bash
git add CLI/sdf_utils.py requirements.txt tests/test_sdf_utils.py
git commit -m "perf: add exact GPU distance transform backend"
```

### Task 5: Ephemeral Boundary Context And BFL Kernel

**Files:**
- Modify: `CLI/advanced_lbm_solver.py`
- Modify: `CLI/d3q27_kernels.py`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Test: `tests/test_advanced_lbm_solver.py`
- Test: `tests/test_d3q27_kernels.py`

- [ ] **Step 1: Write failing context reuse tests**

```python
def test_boundary_context_prepares_once_per_solver_call():
    solver = make_solver(16)
    context = solver.prepare_boundary_context(geometry)
    solver.collide_stream(geometry, steps=5, boundary_context=context)
    assert context.prepare_count == 1
    assert context.step_use_count == 5
```

Add fixed-geometry reference tests for BFL output populations, boundary links,
force accumulators, and objective coefficients.

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_advanced_lbm_solver.py tests/test_d3q27_kernels.py -q`

Expected: missing boundary context and BFL kernel.

- [ ] **Step 3: Implement `GeometryBoundaryContext`**

Prepare SDF, `q`, and boundary links once for an immutable solver geometry.
Pass the context through every LBM step. Remove per-probe CPU content hashing;
the SPSA caller owns context lifetime and releases it after each base/plus/minus
solve.

- [ ] **Step 4: Implement parity-gated Triton BFL kernel**

Fuse the 26 direction updates while preserving both BFL interpolation branches,
non-periodic indexing, opposite writes, and active masks. Keep the PyTorch BFL
reference selectable.

- [ ] **Step 5: Add numerical parity gates**

Compare reference and Triton populations, forces, drag/lift coefficients, and
direct objective on deterministic 8, 16, and 32 grids. Triton activation must
raise or use the explicit reference backend when parity is not established.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_advanced_lbm_solver.py tests/test_d3q27_kernels.py -q`

Expected: all CPU tests and available CUDA parity tests pass.

- [ ] **Step 7: Commit**

```bash
git add CLI/advanced_lbm_solver.py CLI/d3q27_kernels.py CLI/aircraft_diffusion_cfd.py tests/test_advanced_lbm_solver.py tests/test_d3q27_kernels.py
git commit -m "perf: keep BFL boundary preparation on GPU"
```

### Task 6: Loss-Only Training Integration

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/run_with_resource_monitor.py`
- Test: `tests/test_aerodynamic_loss.py`
- Test: `tests/test_overfit_stop.py`

- [ ] **Step 1: Write failing loss-accounting tests**

```python
def test_every_computed_training_term_is_in_optimization_loss():
    terms = measured_training_terms()
    assert set(terms.computed) == set(terms.optimized)
    assert {"aerodynamic", "connectivity", "aircraft_validity"} <= set(terms.optimized)


def test_training_metrics_do_not_publish_zero_diagnostic_placeholders():
    metrics = one_fake_epoch_metrics()
    assert "diagnostic_total" not in metrics
    assert "connectivity" not in metrics
    assert "aerodynamic" not in metrics
    assert metrics["direct_solver_iteration_coverage"] == 1.0
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_aerodynamic_loss.py tests/test_overfit_stop.py -q`

Expected: monitor-only fields and computations remain.

- [ ] **Step 3: Consolidate loss-bearing terms**

The optimization loss contains diffusion MSE, target reconstruction,
conditioned generation reconstruction, consistency, and measured direct SPSA
loss. The direct loss contains CFD aerodynamic, exact connectivity,
aircraft-validity, and occupancy terms. Remove duplicate training-time
connectivity/aerodynamic monitor calls and zero placeholder metrics.

- [ ] **Step 4: Add compact staging and async transfers**

Use pinned compact batches and `non_blocking=True`. Stage at most two batches.
Keep Windows worker count zero. Record transfer and component timings without
performing extra physics evaluations.

- [ ] **Step 5: Extend resource monitoring**

Record process/disk I/O counters and component timings from the training metrics
stream. Monitoring must observe existing work and never invoke solver
diagnostics.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_aerodynamic_loss.py tests/test_overfit_stop.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add CLI/aircraft_diffusion_cfd.py CLI/run_with_resource_monitor.py tests/test_aerodynamic_loss.py tests/test_overfit_stop.py
git commit -m "perf: remove non-optimizing training diagnostics"
```

### Task 7: Full Verification And Controlled Launch

**Files:**
- Create: `docs/benchmarks/g96_expanded_gpu_probe_20260629.md`
- Update generated evidence under ignored `build/`

- [ ] **Step 1: Install/probe the optional CUDA dependency**

Install a CUDA-12/Python-3.12-compatible CuPy wheel in the local environment.
Run the GPU exact EDT startup/parity probe. Record package and CUDA versions.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Run one-iteration `96^3` correctness/memory probe**

Use the expanded model, compact 600-geometry manifest, BF16 model math, FP32
solver, exact GPU EDT, parity-approved BFL backend, and solver interval one.
Abort if peak VRAM exceeds 8,050 MiB or any loss term is absent/non-finite.

- [ ] **Step 4: Run 20-iteration reference/optimized benchmark**

Use identical geometry IDs, conditions, seeds, perturbations, and solver steps.
Compare objective values, force coefficients, gradients, iteration time, CPU,
GPU, RAM, disk I/O, SDF time, BFL time, decoder time, and transfer time.

- [ ] **Step 5: Write the benchmark report**

State exact results and failures. Do not claim speedup, parity, or aircraft
validity unless the recorded artifacts support it.

- [ ] **Step 6: Stop the old run only after probe success**

Allow the current batch/epoch to finish or terminate it through its normal
process boundary after a checkpoint exists. Do not delete its logs or
checkpoints.

- [ ] **Step 7: Launch the fresh until-gated run**

Start the expanded `96^3` run with no wall-clock timeout and resource/component
monitoring. Require 100% per-iteration measured solver coverage and the existing
geometry promotion gate.

- [ ] **Step 8: Commit and push**

```bash
git add docs/benchmarks/g96_expanded_gpu_probe_20260629.md
git commit -m "docs: record expanded GPU training probe"
git push origin HEAD:reviews
```

