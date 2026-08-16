
import logging

import torch
import numpy as np
from concurrent.futures import Future
from typing import Dict, Optional, Sequence
from typing import TYPE_CHECKING

from lbm_utils import (
    D3Q27Lattice,
    REFERENCE_SPEED_OF_SOUND_MPS,
    _compute_force_coefficients,
    build_lbm_compressibility_metadata,
    mach_to_lattice_velocity,
    mach_to_physical_speed,
)
from lbm_diagnostics import compute_strain_rate_tensor, compute_vorticity, compute_velocity_gradients
from sdf_utils import compute_all_link_distances, compute_link_q
from lbm_logger import LBMLogger
from utils import compute_tensor_content_hash

try:
    from d3q27_kernels import stream_bounce_d3q27
except Exception:  # pragma: no cover - optional acceleration path
    stream_bounce_d3q27 = None

try:
    from d3q27_kernels import stream_bfl_d3q27
except Exception:  # pragma: no cover - optional acceleration path
    stream_bfl_d3q27 = None

try:
    from d3q27_kernels import stream_bfl_d3q27_batch
except Exception:  # pragma: no cover - optional acceleration path
    stream_bfl_d3q27_batch = None

try:
    from d3q27_kernels import stream_bfl_d3q27_batch_compressed
except Exception:  # pragma: no cover - optional acceleration path
    stream_bfl_d3q27_batch_compressed = None

# Task 35: bf16 population-storage knob for the batched workspace. Default
# fp32 (the production precision contract). The experiment sets
# ``D3Q27Solver.batch_population_dtype = torch.bfloat16`` to halve the two
# resident batch population buffers (``_f_batch`` / ``_f_swap_batch``). The
# design is "bf16 STORAGE, fp32 COMPUTE": every reduction, the collision
# matmul, the force accumulators, and the stream/BFL kernels upcast bf16 loads
# to fp32 before arithmetic, so the ONLY difference between the fp32 and bf16
# runs is the stored population state, not reduction precision. OFF by default.
_BATCH_POPULATION_DTYPE = torch.float32

try:
    from d3q27_kernels import force_drag_d3q27
except Exception:  # pragma: no cover - optional acceleration path
    force_drag_d3q27 = None


def _scale_momentum_exchange_force(force, grid_spacing: float, mach_number: float, density: float = 1.225):
    """Convert raw lattice momentum exchange into a physical force scale (Issue #16).

    The extracted wall sum is a perturbational lattice momentum flux acting on the
    fluid. Converting it to the body-force convention used by the CFD outputs
    requires:

    1. multiplying by the lattice freestream speed to recover the missing O(U)
       factor needed for force ~ U^2 at fixed Reynolds number.

    Using dt = dx / (REFERENCE_SPEED_OF_SOUND_MPS * sqrt(3)) gives the
    remaining force scale rho_phys * dx^2 * (REFERENCE_SPEED_OF_SOUND_MPS * sqrt(3))^2.
    """
    dx = float(grid_spacing)
    # velocity_ratio = sound_speed_phys / sound_speed_lattice = REFERENCE_SPEED_OF_SOUND_MPS * sqrt(3)
    velocity_ratio = REFERENCE_SPEED_OF_SOUND_MPS * np.sqrt(3.0)
    force_scale = float(density) * (dx**2) * (velocity_ratio**2)
    lattice_freestream_speed = abs(float(mach_number)) / np.sqrt(3.0)
    return lattice_freestream_speed * force * force_scale


class D3Q27Solver:
    """Complete D3Q27 LBM solver with vectorized Cascaded MRT collision operator."""
    def __init__(
        self,
        resolution,
        device,
        inlet_velocity_lu: float = 0.0,
        use_triton_streaming: bool = False,
        use_fused_stream_bfl: bool = False,
    ):
        self.res = resolution
        self.device = device
        self.inlet_velocity_lu = float(inlet_velocity_lu)

        self.ex, self.ey, self.ez = D3Q27Lattice.get_vectors()
        self.ex = self.ex.to(device, dtype=torch.long)
        self.ey = self.ey.to(device, dtype=torch.long)
        self.ez = self.ez.to(device, dtype=torch.long)
        self.ex_f = self.ex.to(dtype=torch.float32).view(-1, 1, 1, 1)
        self.ey_f = self.ey.to(dtype=torch.float32).view(-1, 1, 1, 1)
        self.ez_f = self.ez.to(dtype=torch.float32).view(-1, 1, 1, 1)
        self.w = D3Q27Lattice.get_weights().to(device)
        self.opposite = D3Q27Lattice.get_opposite().to(device)
        self._opposite_list = D3Q27Lattice.get_opposite().tolist()
        self._stream_shifts = [
            (int(dx), int(dy), int(dz))
            for dx, dy, dz in zip(self.ex.tolist(), self.ey.tolist(), self.ez.tolist())
        ]
        self._force_dirs = [i for i in range(27) if i != 0]
        self._force_dir_index = torch.tensor(self._force_dirs, dtype=torch.long, device=device)
        self._force_ex = self.ex[self._force_dir_index].to(dtype=torch.float32).view(-1, 1, 1, 1)
        self._force_ez = self.ez[self._force_dir_index].to(dtype=torch.float32).view(-1, 1, 1, 1)
        self._force_speed = torch.sqrt(
            self.ex[self._force_dir_index].to(dtype=torch.float32)**2
            + self.ey[self._force_dir_index].to(dtype=torch.float32)**2
            + self.ez[self._force_dir_index].to(dtype=torch.float32)**2
        ).view(-1, 1, 1, 1)

        # Task 8 #1: precomputed broadcast factors for the vectorized 26-dir
        # momentum-exchange accumulation. Distinct from _opposite_list /
        # _force_ex above: these are tensor views over links 1..26 shaped for
        # a single [26, D, H, W] broadcast reduction.
        self._force_ex_links = self.ex[1:27].float().view(26, 1, 1, 1)
        self._force_ez_links = self.ez[1:27].float().view(26, 1, 1, 1)
        self._opposite_links = torch.tensor(
            self._opposite_list[1:27], dtype=torch.long, device=device
        )

        # Precompute moment basis for vectorized MRT Collision (Issue #16)
        self.moment_keys = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        basis_rows = []
        ex_f_vec = self.ex.to(dtype=torch.float32)
        ey_f_vec = self.ey.to(dtype=torch.float32)
        ez_f_vec = self.ez.to(dtype=torch.float32)
        for a, b, c in self.moment_keys:
            basis_rows.append((ex_f_vec ** a) * (ey_f_vec ** b) * (ez_f_vec ** c))
        self.moment_basis = torch.stack(basis_rows, dim=0).to(device)
        self.moment_basis_inv = torch.inverse(self.moment_basis)

        # Precompute relaxation indices for MRT
        # 0: Conserved, 1: Energy, 2: Shear, 3: High-order
        self.s_indices = torch.zeros(27, dtype=torch.long, device=device)
        self.conserved_indices = []
        for i, (a, b, c) in enumerate(self.moment_keys):
            # Conserved moments: rho(000), jx(100), jy(010), jz(001)
            if (a == 0 and b == 0 and c == 0) or (a == 1 and b == 0 and c == 0) or \
               (a == 0 and b == 1 and c == 0) or (a == 0 and b == 0 and c == 1):
                self.s_indices[i] = 0
                self.conserved_indices.append(i)
            elif (a, b, c) in [(2, 0, 0), (0, 2, 0), (0, 0, 2)]:
                self.s_indices[i] = 1  # Energy
            elif (a, b, c) in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]:
                self.s_indices[i] = 2  # Shear (determines viscosity)
            else:
                self.s_indices[i] = 3  # High-order

        self.conserved_indices = torch.tensor(self.conserved_indices, dtype=torch.long, device=device)
        self.s_e = 1.2
        self.s_h = 1.6

        # Cache masks for vectorized boundary population updates
        self._inlet_mask = self.ex > 0
        self._outlet_mask = self.ex < 0

        self.drag_link_metric_exponent = None
        self._boundary_cache_key = None
        self._boundary_link_cache = None
        self.use_triton_streaming = bool(use_triton_streaming and stream_bounce_d3q27 is not None and device.type == "cuda")
        # Fused pull-stream + BFL backend (parity-gated). Unsupported
        # environments fall back explicitly to the PyTorch reference path and
        # never to the simplified bounce kernel above.
        fused_requested = bool(use_fused_stream_bfl)
        self.use_fused_stream_bfl = bool(
            fused_requested and stream_bfl_d3q27 is not None and device.type == "cuda"
        )
        self._fused_stream_bfl_fallback_warned = not fused_requested or self.use_fused_stream_bfl
        # Relaxation vectors are constant for a fixed omega; rebuilding them
        # every step created one CUDA tensor per step (plan Phase 1).
        self._s_vec_cache = {}
        # P6h: deterministic per-block partial accumulator for the fused
        # momentum-exchange force + projected-drag kernel, keyed by block count
        # (fixed for a given resolution). Each slot is written exactly once per
        # call, then reduced with one torch.sum.
        self._force_drag_partials_cache = {}

        # 27 populations
        self.f = torch.zeros(27, resolution, resolution, resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)

        # Macroscopic fields stored for diagnostics/synchronization
        self.velocity_x = torch.zeros(resolution, resolution, resolution, device=device)
        self.velocity_y = torch.zeros_like(self.velocity_x)
        self.velocity_z = torch.zeros_like(self.velocity_x)
        self.pressure = torch.zeros_like(self.velocity_x)
        self.rho = torch.ones_like(self.velocity_x)

        self.reset_force_accounting()
        self.logger = LBMLogger()

        # Cache for BFL distances keyed by geometry hash (Fix A)
        self._q_cache = {}

        # Task 9: pre-warmed SDF/q cache keyed by the same geometry hash. Each
        # entry is a concurrent.futures.Future that produces a CPU q tensor (or,
        # after the future resolves, the tensor itself once _get_q pops it). The
        # SPSA probe pre-warm in aircraft_diffusion_cfd fills this so the CPU EDT
        # runs on a thread pool in parallel with the GPU LBM solves. Entries stay
        # on CPU; _get_q pops one and moves it to the solve device.
        self._warm_sdf_cache = {}

        # Task 10: private batched-path workspaces. These are NEVER touched by
        # the sequential path (collide_and_stream/_get_q/_boundary_links) and
        # never alias the single-geometry _q_cache / _boundary_link_cache /
        # drag_link_metric_exponent / self.f. They are allocated lazily by
        # _init_batch_equilibrium at the batch size of the current call.
        # Task 35: the population-buffer dtype is a knob, default fp32. Setting
        # this attribute to torch.bfloat16 stores the batch populations in bf16
        # (halving the two resident buffers); all arithmetic stays fp32.
        self.batch_population_dtype = _BATCH_POPULATION_DTYPE
        self._f_batch = None
        self._f_pre_batch = None
        self._f_temp_batch = None
        self._f_swap_batch = None
        # Task 34: compact boundary-link (active-voxel) tables keyed by
        # (geom_hash tuple, C, res). Only boundary q crosses to GPU; the
        # full-lattice [C, 27, D, H, W] q-field is never resident here.
        self._bfl_sparse_cache = {}
        self._velocity_x_batch = None
        self._velocity_y_batch = None
        self._velocity_z_batch = None
        self._pressure_batch = None
        self._rho_batch = None
        self._force_x_accum_batch = None
        self._force_z_accum_batch = None
        self._projected_drag_accum_batch = None
        self._force_samples_batch = None
        self._force_x_last_batch = None
        self._force_z_last_batch = None
        self._projected_drag_last_batch = None
        self._force_sample_start_batch = 0
        self._force_step_batch = 0

        # Precompute Guo directions
        self.ei_guo = torch.stack([self.ex_f, self.ey_f, self.ez_f], dim=1).view(27, 3, 1, 1, 1)

    def _materialize_q_from_warm_entry(self, entry, geometry_mask):
        """Materialize an on-device q tensor from a warm-pool entry.

        Shared by the sequential ``_get_q`` and the Task 10 batched helpers
        (``_get_q_batch`` / ``_get_q_single_batch``). OFFLOAD-3: the pool now
        pre-computes a [D, H, W] SDF (scipy EDT stays on CPU); the 26-link
        q-algebra runs on the solve device, so only the small SDF crosses H2D
        instead of the full [27, D, H, W] q field. A full 4-D q tensor (legacy
        warm entries / unit-test sentinels) is still used as-is.
        """
        if entry.ndim == 3:
            return compute_link_q(
                entry.to(geometry_mask.device), self.ex, self.ey, self.ez
            )
        return entry.to(geometry_mask.device)

    def _get_q(self, geometry_mask, geom_hash=None):
        """Get or compute sub-voxel distances for the given geometry (Fix A/Issue #15)."""
        # Fix A: Use a true content hash for the geometry key (Review Feedback)
        # Issue #22: Support pre-computed hash to reduce CPU-GPU sync overhead
        geom_key = geom_hash if geom_hash is not None else compute_tensor_content_hash(geometry_mask)

        if geom_key not in self._q_cache:
            # Task 9: a warm entry is a pre-computed CPU tensor, or an in-flight
            # Future that will produce one, submitted by the SPSA probe pre-warm.
            # Pop it, materialize it on the solve device (see
            # _materialize_q_from_warm_entry), and store it in the per-solve
            # _q_cache so the 5 solver steps reuse it. On a miss we fall back to
            # the original compute-and-store path (the EDT then runs serially).
            warm_entry = self._warm_sdf_cache.pop(geom_key, None)
            if warm_entry is not None:
                entry = warm_entry.result() if isinstance(warm_entry, Future) else warm_entry
                self._q_cache[geom_key] = self._materialize_q_from_warm_entry(entry, geometry_mask)
                # Keep the pre-warm pool topped up (bounded in-flight so the CPU
                # does not accumulate all 33 q tensors at once).
                refill = getattr(self, "_sdf_refill", None)
                if callable(refill):
                    refill(self)
            else:
                # CPU/SciPy cost is explicit here.
                self._q_cache[geom_key] = compute_all_link_distances(geometry_mask, self.ex, self.ey, self.ez)

        # Optional: Limit cache size to prevent OOM
        if len(self._q_cache) > 100:
            self._q_cache.pop(next(iter(self._q_cache)))

        return self._q_cache[geom_key]

    def compute_equilibrium(self, rho, ux, uy, uz):
        cu = self.ex_f * ux + self.ey_f * uy + self.ez_f * uz
        u_sq = ux**2 + uy**2 + uz**2
        return self.w.view(-1, 1, 1, 1) * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

    def compute_moment_equilibrium(self, rho, ux, uy, uz):
        """Equilibrium tensor-product moments for D3Q27 (Issue #16, Task 8 #3).

        The 27-loop is fused into a [3,3,3,D,H,W] tensor-product broadcast and
        reshaped in the same a*9+b*3+c order as ``moment_keys``. The per-element
        operand order ``((rho * mx[a]) * my[b]) * mz[c]`` is unchanged, so the
        result is bitwise identical to the loop (``torch.equal`` verifiable).
        """
        cs2 = 1.0 / 3.0
        mx = torch.stack([torch.ones_like(rho), ux, ux * ux + cs2])
        my = torch.stack([torch.ones_like(rho), uy, uy * uy + cs2])
        mz = torch.stack([torch.ones_like(rho), uz, uz * uz + cs2])
        meq = rho * mx[:, None, None] * my[None, :, None] * mz[None, None, :]
        return meq.reshape(27, *rho.shape)

    def reset_force_accounting(self, sample_start: int = 0):
        """Reset momentum-exchange bookkeeping for a new simulation run."""
        self.force_x_accum = torch.tensor(0.0, device=self.device)
        self.force_z_accum = torch.tensor(0.0, device=self.device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=self.device)
        self.force_z_last = torch.tensor(0.0, device=self.device)
        self.projected_drag_accum = torch.tensor(0.0, device=self.device)
        self.projected_drag_last = torch.tensor(0.0, device=self.device)
        self._force_sample_start = max(0, int(sample_start))
        self._force_step = 0

    def _accumulate_momentum_exchange_force(self, geometry_mask, geom_hash=None):
        """Compute wall force from fluid-solid links using bounce-back exchange and BFL correction."""
        # Task 8 #1: the fused broadcast below keeps this path identical to the
        # nosync variant (sums over empty selections contribute 0.0), so the
        # reference path shares the same kernel and parity envelope.
        return self._accumulate_momentum_exchange_force_nosync(geometry_mask, geom_hash=geom_hash)

    def _accumulate_momentum_exchange_force_nosync(self, geometry_mask, geom_hash=None):
        """Vectorized 26-dir momentum-exchange sum (Task 8 #1).

        The 26-direction loop is fused into three [26, D, H, W] broadcasts:
        f_in from pre-stream, f_out from the opposite link, then a single
        reduction over the weighted sum. Reduction order differs from the old
        per-direction loop (~1e-13 relative, LOW parity), so the fused-parity
        gate (FORCE_ATOL 2.5e-5) is the contract rather than bitwise equality.
        """
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)

        f_in_all = self.f_pre_stream[1:27] * boundary_links
        f_out_all = self.f_temp[self._opposite_links] * boundary_links
        sum_all = f_in_all + f_out_all
        step_force_x = (self._force_ex_links * sum_all).sum()
        step_force_z = (self._force_ez_links * sum_all).sum()
        return step_force_x, step_force_z

    def _effective_drag_link_metric_exponent(self, geometry_mask):
        if self.drag_link_metric_exponent is not None:
            return float(self.drag_link_metric_exponent)
        projected_cells = torch.sum(torch.any(geometry_mask > 0.5, dim=0).float()).item()
        projected_side = float(np.sqrt(max(projected_cells, 1.0)))
        return float(np.clip(1.68 - 0.295 * (projected_side - 13.0), 0.5, 1.68))

    def _accumulate_projected_pressure_drag_proxy(self, geometry_mask, geom_hash=None):
        """Coarse-grid pressure-drag proxy from upwind-facing D3Q27 wall links."""
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)
        flow_sign = 1.0 if self.inlet_velocity_lu >= 0.0 else -1.0
        upwind = (self._force_ex * flow_sign) > 0.0
        metric_exponent = self._effective_drag_link_metric_exponent(geometry_mask)
        metric = torch.pow(self._force_speed.clamp_min(1.0), -metric_exponent)
        projected = 2.0 * torch.abs(self._force_ex) * metric * self.f_pre_stream[self._force_dir_index]
        return torch.sum(torch.where(upwind, projected * boundary_links, torch.zeros_like(projected)))

    def _accumulate_force_drag_fused(self, geometry_mask, geom_hash=None, block_size: int = 256):
        """Fused momentum-exchange force + projected drag (P6h FUSION-3).

        Runs one Triton pass over the 26 wall links that accumulates fx, fz and
        the projected-drag proxy into a deterministic per-block partial buffer,
        then a single small ``torch.sum`` reduces it. Replaces the two separate
        full-array reductions (``_accumulate_momentum_exchange_force_nosync`` +
        ``_accumulate_projected_pressure_drag_proxy``) on the fused stream path.
        The per-element formulas and operand order match the reference exactly;
        the only drift is the block-partitioned reduction tree (~1e-7 relative
        to the gross on the 32^3 parity fixtures, worst 5-step signed drift
        ~1.5e-5 absolute, well within FORCE_ATOL 2.5e-5). Keeps fp32 throughout.
        """
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)
        total = self.f_pre_stream[0].numel()
        n_blocks = (total + block_size - 1) // block_size
        partials = self._force_drag_partials_cache.get(n_blocks)
        if partials is None:
            partials = torch.zeros(3, 26, n_blocks, device=self.device, dtype=torch.float32)
            self._force_drag_partials_cache[n_blocks] = partials

        flow_sign = 1.0 if self.inlet_velocity_lu >= 0.0 else -1.0
        upwind = ((self._force_ex * flow_sign) > 0.0).float().reshape(26).contiguous()
        metric_exponent = self._effective_drag_link_metric_exponent(geometry_mask)
        metric = torch.pow(self._force_speed.clamp_min(1.0), -metric_exponent)
        drag_weight = (2.0 * torch.abs(self._force_ex) * metric).reshape(26).contiguous()

        ok = force_drag_d3q27(
            self.f_pre_stream,
            self.f_temp,
            boundary_links,
            self._force_ex_links.reshape(26).contiguous(),
            self._force_ez_links.reshape(26).contiguous(),
            self.opposite,
            drag_weight,
            upwind,
            partials,
            block_size=block_size,
        )
        if not ok:
            step_force_x, step_force_z = self._accumulate_momentum_exchange_force_nosync(
                geometry_mask, geom_hash=geom_hash
            )
            step_projected_drag = self._accumulate_projected_pressure_drag_proxy(
                geometry_mask, geom_hash=geom_hash
            )
            return step_force_x, step_force_z, step_projected_drag

        sums = torch.sum(partials, dim=(-2, -1))
        return sums[0], sums[1], sums[2]

    def _boundary_links(self, geometry_mask, geom_hash=None):
        """Cache static fluid-solid links without boundary wraparound (Issue #15)."""
        # Use true content hash for cache key (Review Feedback)
        # Issue #22: Support pre-computed hash to reduce CPU-GPU sync overhead
        cache_key = geom_hash if geom_hash is not None else compute_tensor_content_hash(geometry_mask)
        if cache_key == self._boundary_cache_key and self._boundary_link_cache is not None:
            return self._boundary_link_cache

        mask = geometry_mask > 0.5

        links = []
        fluid = ~mask
        D, H, W = mask.shape
        # Pad mask to detect neighbors without wraparound
        mask_padded = torch.nn.functional.pad(mask, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        for i in self._force_dirs:
            dx, dy, dz = self._stream_shifts[i]
            # Neighbor at (x+dx, y+dy, z+dz).
            # In mask_padded, this corresponds to (1+x+dx, 1+y+dy, 1+z+dz)
            d_s, d_e = 1+dx, 1+dx+D
            h_s, h_e = 1+dy, 1+dy+H
            w_s, w_e = 1+dz, 1+dz+W
            neighbor_is_solid = mask_padded[d_s:d_e, h_s:h_e, w_s:w_e]
            links.append(fluid & neighbor_is_solid)

        self._boundary_link_cache = torch.stack(links, dim=0)
        self._boundary_cache_key = cache_key
        return self._boundary_link_cache

    def _apply_domain_boundaries(self):
        """Apply vectorized non-periodic domain boundaries (Equilibrium Inlet, Neumann Outlet)."""
        # Equilibrium Inlet at X=0
        if self.inlet_velocity_lu != 0.0:
            inlet_shape = self.f[:, 0, :, :].shape[1:]
            rho_in = torch.ones(inlet_shape, device=self.device, dtype=self.f.dtype)
            ux_in = torch.full_like(rho_in, self.inlet_velocity_lu)
            uy_in = torch.zeros_like(rho_in)
            uz_in = torch.zeros_like(rho_in)

            cu = (
                self.ex.to(dtype=self.f.dtype).view(-1, 1, 1) * ux_in
                + self.ey.to(dtype=self.f.dtype).view(-1, 1, 1) * uy_in
                + self.ez.to(dtype=self.f.dtype).view(-1, 1, 1) * uz_in
            )
            u_sq = ux_in**2 + uy_in**2 + uz_in**2
            feq_in = self.w.to(dtype=self.f.dtype).view(-1, 1, 1) * rho_in * (
                1 + 3 * cu + 4.5 * cu**2 - 1.5 * u_sq
            )
            # Vectorized inlet update: overwrite populations streaming INTO the domain
            self.f_temp[self._inlet_mask, 0, :, :] = feq_in[self._inlet_mask]

        # Vectorized Neumann (Zero-Gradient) Outlet at X=-1 (Issue #16 fix)
        # Use interior plane after streaming but before BC for true zero-gradient
        self.f_temp[self._outlet_mask, -1, :, :] = self.f_temp[self._outlet_mask, -2, :, :]

        # Slip Walls (Mirror) or Neumann for other boundaries - Vectorized slices
        self.f_temp[:, :, 0, :] = self.f_temp[:, :, 1, :]
        self.f_temp[:, :, -1, :] = self.f_temp[:, :, -2, :]
        self.f_temp[:, :, :, 0] = self.f_temp[:, :, :, 1]
        self.f_temp[:, :, :, -1] = self.f_temp[:, :, :, -2]

    def _apply_bfl_boundary(self, geometry_mask, geom_hash=None):
        """Bouzidi-Firdaouss-Lallemand (BFL) boundary condition."""
        q = self._get_q(geometry_mask, geom_hash=geom_hash)

        # Standard bounce-back links
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)

        # Issue #22: Optimize padding by padding the entire tensor once if possible,
        # but since directions vary, we still need to shift.
        # We can at least pad the spatial dimensions of f_pre_stream once.
        # self.f_pre_stream: [27, D, H, W]
        D, H, W = geometry_mask.shape
        f_pre_padded = torch.nn.functional.pad(self.f_pre_stream, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        for i in range(1, 27):
            opp_i = self._opposite_list[i]
            link_idx = i - 1

            # Identify boundary links
            active = boundary_links[link_idx]
            if not torch.any(active):
                continue

            qi = q[i][active]

            # Identify neighbors without wraparound (Issue #15)
            dx, dy, dz = self._stream_shifts[i]

            # Neighbor at x-e_i is at (1-dx, 1-dy, 1-dz) in padded coords
            # f_pre_padded has shape [27, D+2, H+2, W+2]
            f_neighbor = f_pre_padded[i, 1-dx:1-dx+D, 1-dy:1-dy+H, 1-dz:1-dz+W][active]

            # BFL interpolation based on q
            q_low = qi < 0.5
            q_high = ~q_low

            # Simplified BFL Linear Interpolation
            # f_opp(x, t+1) = (1-2q)f_i(x, t) + 2q f_i(x, t)_bb (if q < 0.5)
            # f_opp(x, t+1) = (1 - 1/(2q))f_opp(x, t) + (1/(2q))f_i(x, t) (if q >= 0.5)

            res = torch.zeros_like(qi)

            # q < 0.5
            if torch.any(q_low):
                res[q_low] = (1 - 2*qi[q_low]) * f_neighbor[q_low] + 2*qi[q_low] * self.f_pre_stream[i][active][q_low]

            # q >= 0.5
            if torch.any(q_high):
                res[q_high] = (1 / (2*qi[q_high])) * self.f_pre_stream[i][active][q_high] + (1 - 1/(2*qi[q_high])) * self.f_temp[opp_i][active][q_high]

            self.f_temp[opp_i][active] = res

    def _compute_guo_forcing(self, rho, u, F, omega):
        """Guo's forcing source term."""
        # F is the external force field [3, D, H, W]
        # u is macroscopic velocity
        # omega is 1/tau
        cs2 = 1.0 / 3.0

        # Precompute u.F
        uF = torch.sum(u * F, dim=0)

        # Factor (1 - 1/(2*tau)) = (1 - omega/2)
        factor = (1.0 - 0.5 * omega)

        # Vectorized Guo source term
        # S_i = w_i * factor * [ (e_i - u)/cs2 + (e_i . u)*e_i / cs2^2 ] . F

        # self.ei_guo: [27, 3, 1, 1, 1]
        # u: [3, D, H, W]
        # F: [3, D, H, W]

        # e_i . u: [27, D, H, W]
        ei_u = torch.sum(self.ei_guo * u.unsqueeze(0), dim=1)
        # e_i . F: [27, D, H, W]
        ei_F = torch.sum(self.ei_guo * F.unsqueeze(0), dim=1)

        term1 = (ei_F - uF) / cs2
        term2 = (ei_u * ei_F) / (cs2**2)

        S = factor * self.w.view(27, 1, 1, 1) * (term1 + term2)
        return S

    def collide_and_stream(self, omega, geometry_mask, ext_force=None, geom_hash=None):
        geometry_mask = geometry_mask.to(self.device, non_blocking=True)
        # Guard against runaway non-finite populations from previous steps.
        self.f.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        # P6e FUSION-2: derive the conserved macroscopic fields (rho / jx / jy
        # / jz) from the moment-matrix rows K[0]/K[9]/K[3]/K[1] of the MRT
        # moment projection instead of four separate full-array reductions. The
        # tensordot is needed for the collision anyway, so the rho/ux/uy/uz
        # reads fold into its single pass (accepted LOW-parity reduction order;
        # the conserved rows of K_post are re-imposed from Keq below, so exact
        # conservation is unchanged).
        K = torch.tensordot(self.moment_basis, self.f, dims=([1], [0]))
        rho = K[0].clamp_min(1e-8)

        # Macroscopic velocity with forcing offset (Guo's definition)
        # u = (sum fi ci + 0.5 F) / rho
        # The direct-training path passes no external force; a host boolean
        # avoids allocating and reducing a 3 x N^3 zero field every step
        # (plan Phase 1).
        has_ext_force = ext_force is not None

        ux_raw = K[9]
        uy_raw = K[3]
        uz_raw = K[1]

        if has_ext_force:
            ux = (ux_raw + 0.5 * ext_force[0]) / (rho + 1e-12)
            uy = (uy_raw + 0.5 * ext_force[1]) / (rho + 1e-12)
            uz = (uz_raw + 0.5 * ext_force[2]) / (rho + 1e-12)
        else:
            ux = ux_raw / (rho + 1e-12)
            uy = uy_raw / (rho + 1e-12)
            uz = uz_raw / (rho + 1e-12)

        ux = ux.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        uy = uy.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        uz = uz.nan_to_num(0.0, posinf=0.0, neginf=0.0)

        # Cascaded MRT Collision (Issue #16)
        Keq = self.compute_moment_equilibrium(rho, ux, uy, uz)

        # Build S-vector (relaxation rates); cached per omega (plan Phase 1).
        s_key = (float(omega), float(self.s_e), float(self.s_h))
        S = self._s_vec_cache.get(s_key)
        if S is None:
            s_vec = torch.tensor([0.0, self.s_e, float(omega), self.s_h], device=self.device)
            S = s_vec[self.s_indices].view(27, 1, 1, 1)
            if len(self._s_vec_cache) > 32:
                self._s_vec_cache.clear()
            self._s_vec_cache[s_key] = S

        # MRT relaxation towards equilibrium
        K_post = K + S * (Keq - K)

        # Actually apply Guo's forcing if ext_force is non-zero
        if has_ext_force and torch.any(ext_force != 0):
            S_guo = self._compute_guo_forcing(rho, torch.stack([ux, uy, uz]), ext_force, omega)
            # Transform Guo source term to moment space if using MRT,
            # or just add to f if using SRT.
            # For MRT, it's easier to add it to populations AFTER MRT relaxation.
            # fi = fi + Si
            # But we are in moment space K_post. So we transform Si to moment space.
            K_S = torch.tensordot(self.moment_basis, S_guo, dims=([1], [0]))
            K_post += K_S

        # Enforce exact conservation of mass and momentum (Issue #16 fix)
        K_post[self.conserved_indices] = Keq[self.conserved_indices]

        # Transform back to populations using in-place copy
        self.f.copy_(torch.tensordot(self.moment_basis_inv, K_post, dims=([1], [0])))
        self.f.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        self.f_pre_stream.copy_(self.f)
        
        used_triton = False
        if self.use_triton_streaming:
            used_triton = stream_bounce_d3q27(
                self.f,
                self.f_pre_stream,
                self.f_temp,
                geometry_mask,
                self.ex,
                self.ey,
                self.ez,
                self.opposite,
            )

        used_fused_bfl = False
        if not used_triton and self.use_fused_stream_bfl and stream_bfl_d3q27 is not None:
            q = self._get_q(geometry_mask, geom_hash=geom_hash)
            solid_u8 = (geometry_mask > 0.5).to(torch.uint8).contiguous()
            used_fused_bfl = stream_bfl_d3q27(
                self.f_pre_stream,
                self.f_temp,
                solid_u8,
                q.contiguous(),
                self.ex,
                self.ey,
                self.ez,
                self.opposite,
            )
            if not used_fused_bfl and not self._fused_stream_bfl_fallback_warned:
                self.logger.log_warning(
                    "Fused stream/BFL kernel unavailable at dispatch; using the "
                    "pytorch_reference streaming and BFL path."
                )
                self._fused_stream_bfl_fallback_warned = True

        if not used_triton and not used_fused_bfl:
            # Streaming
            for i in range(27):
                self.f_temp[i] = torch.roll(self.f[i], shifts=self._stream_shifts[i], dims=(0,1,2))

            # Sub-voxel boundary conditions
            self._apply_bfl_boundary(geometry_mask, geom_hash=geom_hash)

        self._apply_domain_boundaries()

        if self.use_fused_stream_bfl and used_fused_bfl and force_drag_d3q27 is not None:
            step_force_x, step_force_z, step_projected_drag = self._accumulate_force_drag_fused(
                geometry_mask, geom_hash=geom_hash
            )
        else:
            step_force_x, step_force_z = self._accumulate_momentum_exchange_force_nosync(
                geometry_mask, geom_hash=geom_hash
            )
            step_projected_drag = self._accumulate_projected_pressure_drag_proxy(geometry_mask, geom_hash=geom_hash)

        self.f.copy_(self.f_temp)
        # The fused stream kernel clamps its writes (P6h nan_to_num fusion), so
        # the post-stream full-array guard is redundant on the fused path. The
        # fallback path keeps the explicit guard; the guard is never removed.
        if not used_fused_bfl:
            self.f.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        # P6e: collapse the dead post-collision velocity recompute. This method
        # returns the PRE-collision macroscopic fields computed above, and
        # D3Q27CascadedSolver refreshes its own velocity/pressure/rho from those
        # returned values every step; the four post-stream full-array reductions
        # written only to these attributes were consumed by no hot-path reader.
        # Store the pre-collision fields so the attributes stay finite and
        # meaningful for any external (e.g. diagnostics) reader.
        self.velocity_x = ux
        self.velocity_y = uy
        self.velocity_z = uz
        self.pressure = rho * (1.0 / 3.0)
        self.rho = rho

        self.force_x_last = step_force_x
        self.force_z_last = step_force_z
        self.projected_drag_last = step_projected_drag
        if self._force_step >= self._force_sample_start:
            self.force_x_accum += step_force_x
            self.force_z_accum += step_force_z
            self.projected_drag_accum += step_projected_drag
            self.force_samples += 1
        self._force_step += 1
        return ux, uy, uz, rho

    # ------------------------------------------------------------------
    # Task 10: batched SPSA probe path. All workspaces are private to the
    # batched methods; the sequential path is authoritative and untouched.
    # ------------------------------------------------------------------
    def _get_q_batch(self, geom_hashes, geometry_masks):
        """Pop warm q tensors for a batch of geometries and stack them.

        Mirrors the sequential ``_get_q`` pop/cold-fallback/refill-hook for each
        item but returns a ``[C, 27, D, H, W]`` stack owned by the batched path.
        Never writes into the single-geometry ``_q_cache``.
        """
        q_list = []
        for geom_key, geometry_mask in zip(geom_hashes, geometry_masks):
            warm_entry = self._warm_sdf_cache.pop(geom_key, None)
            if warm_entry is not None:
                entry = warm_entry.result() if isinstance(warm_entry, Future) else warm_entry
                q = self._materialize_q_from_warm_entry(entry, geometry_mask)
                refill = getattr(self, "_sdf_refill", None)
                if callable(refill):
                    refill(self)
            else:
                q = compute_all_link_distances(geometry_mask, self.ex, self.ey, self.ez)
            q_list.append(q)
        return torch.stack(q_list, dim=0).contiguous()

    def _get_q_single_batch(self, geometry_mask, geom_hash):
        """Pop/compute the full ``[27, D, H, W]`` q for one batch geometry.

        Same per-geometry body as ``_get_q_batch`` (warm-cache pop or cold EDT
        fallback, refill hook) but returns a single full-lattice tensor. The
        Task 34 sparse-table builder extracts boundary-link q from this and lets
        the full tensor be freed immediately, so no ``[C, 27, D, H, W]`` stack
        is ever resident on GPU.
        """
        warm_entry = self._warm_sdf_cache.pop(geom_hash, None)
        if warm_entry is not None:
            entry = warm_entry.result() if isinstance(warm_entry, Future) else warm_entry
            q = self._materialize_q_from_warm_entry(entry, geometry_mask)
            refill = getattr(self, "_sdf_refill", None)
            if callable(refill):
                refill(self)
        else:
            q = compute_all_link_distances(geometry_mask, self.ex, self.ey, self.ez)
        return q

    def _build_bfl_sparse_tables(self, mask_stack, geom_hashes, q_stack=None, boundary_links_batch=None):
        """Build compact active-voxel (boundary-link) q tables for the batch.

        The fused BFL kernel only ever consumes q at fluid voxels whose in-domain
        neighbor in the incoming direction is solid — exactly
        ``_boundary_links_batch``. For each (c, i) pair this extracts the flat
        voxel offsets and the q values at those voxels into a compact
        concatenation:

            q_flat       [N_active] fp32  q values at active voxels
            active_flat  [N_active] int32 flat voxel offsets
            pair_start   [C*26]     int32 start index per (c, i) pair
            pair_count   [C*26]     int32 active-voxel count per (c, i) pair

        Pair ``p`` is item ``p // 26``, direction ``p % 26 + 1``. The values are
        bit-identical to today's ``q_field[i]`` at active voxels. Tables are
        geometry-static and cached keyed by ``(geom_hashes, C, res)`` so the
        per-step solve loop never rebuilds them. ``q_stack`` (a caller-owned full
        stack) is honored when supplied; otherwise q is computed per geometry and
        the full tensor is freed immediately (only boundary q crosses to GPU).
        """
        C = mask_stack.shape[0]
        res = int(mask_stack.shape[1])
        key = (tuple(geom_hashes), C, res)
        cached = self._bfl_sparse_cache.get(key)
        if cached is not None:
            return cached
        if boundary_links_batch is None:
            boundary_links_batch = self._boundary_links_batch(mask_stack, geom_hashes)
        N = res * res * res
        starts = []
        counts = []
        offs_list = []
        q_list = []
        cum = 0
        for c in range(C):
            if q_stack is not None:
                q_c_flat = q_stack[c].reshape(27, N)
            else:
                q_c = self._get_q_single_batch(mask_stack[c], geom_hashes[c])
                q_c_flat = q_c.reshape(27, N)
            for i in range(1, 27):
                link_idx = i - 1
                idx = boundary_links_batch[c, link_idx].reshape(-1).nonzero(as_tuple=False).reshape(-1)
                n_i = idx.numel()
                starts.append(cum)
                counts.append(n_i)
                if n_i:
                    offs_list.append(idx.to(torch.int32))
                    q_list.append(q_c_flat[i][idx])
                    cum += n_i
            if q_stack is None:
                del q_c
        if offs_list:
            active_flat = torch.cat(offs_list, dim=0).contiguous()
            q_flat = torch.cat(q_list, dim=0).contiguous()
        else:
            active_flat = torch.zeros(0, dtype=torch.int32, device=self.device)
            q_flat = torch.zeros(0, dtype=torch.float32, device=self.device)
        pair_start = torch.tensor(starts, dtype=torch.int32, device=self.device)
        pair_count = torch.tensor(counts, dtype=torch.int32, device=self.device)
        sparse = {
            "active_flat": active_flat,
            "q_flat": q_flat,
            "pair_start": pair_start,
            "pair_count": pair_count,
        }
        if len(self._bfl_sparse_cache) > 64:
            self._bfl_sparse_cache.clear()
        self._bfl_sparse_cache[key] = sparse
        return sparse

    def compute_moment_equilibrium_batch(self, rho, ux, uy, uz):
        """Batched equilibrium tensor-product moments for D3Q27.

        Same per-element operand order ``((rho * mx) * my) * mz`` as
        ``compute_moment_equilibrium``, reshaped to ``[C, 27, D, H, W]``. The
        broadcast shape differs from the single-geometry ``[27, D, H, W]``
        (accepted LOW-parity source for the batched path).
        """
        cs2 = 1.0 / 3.0
        mx = torch.stack([torch.ones_like(rho), ux, ux * ux + cs2], dim=1)  # [C,3,D,H,W]
        my = torch.stack([torch.ones_like(rho), uy, uy * uy + cs2], dim=1)
        mz = torch.stack([torch.ones_like(rho), uz, uz * uz + cs2], dim=1)
        # Build all factors as [C,3,3,3,D,H,W] broadcast sources with the
        # a/b/c index dims in positions 1/2/3, matching the sequential
        # ((rho * mx[a]) * my[b]) * mz[c] operand order per element.
        rho_7 = rho.unsqueeze(1).unsqueeze(2).unsqueeze(3)          # [C,1,1,1,D,H,W]
        mx_7 = mx.unsqueeze(2).unsqueeze(3)                          # [C,3,1,1,D,H,W]
        my_7 = my.unsqueeze(1).unsqueeze(3)                          # [C,1,3,1,D,H,W]
        mz_7 = mz.unsqueeze(1).unsqueeze(2)                          # [C,1,1,3,D,H,W]
        meq = rho_7 * mx_7 * my_7 * mz_7                             # [C,3,3,3,D,H,W]
        return meq.reshape(meq.shape[0], 27, *rho.shape[1:])

    def reset_force_accounting_batch(self, C, sample_start: int = 0):
        """Reset [C]-shaped force bookkeeping for a batched solve."""
        self._force_x_accum_batch = torch.zeros(C, device=self.device)
        self._force_z_accum_batch = torch.zeros(C, device=self.device)
        self._projected_drag_accum_batch = torch.zeros(C, device=self.device)
        self._force_samples_batch = torch.zeros(C, dtype=torch.int64, device=self.device)
        self._force_x_last_batch = torch.zeros(C, device=self.device)
        self._force_z_last_batch = torch.zeros(C, device=self.device)
        self._projected_drag_last_batch = torch.zeros(C, device=self.device)
        self._force_sample_start_batch = max(0, int(sample_start))
        self._force_step_batch = 0

    def _init_batch_equilibrium(self, C):
        """Allocate (if needed) and initialize the private batch buffers to the
        same equilibrium state ``_initialize_equilibrium`` uses for a single
        solve, so every item starts byte-identically to the sequential path.

        Task 34: only two live population buffers (ping-pong roles). ``_f_batch``
        is the current state at step start (collide in-place, then the pre-stream
        source); ``_f_swap_batch`` is the stream destination (post-stream state,
        becomes the next current after the swap). The legacy ``_f_pre_batch`` /
        ``_f_temp_batch`` are retained as None so old external cleanup paths that
        reference them stay harmless.
        """
        if (
            self._f_batch is None
            or self._f_batch.shape[0] != C
            or self._f_batch.dtype != self.batch_population_dtype
        ):
            self._f_batch = torch.empty(
                C,
                27,
                self.res,
                self.res,
                self.res,
                device=self.device,
                dtype=self.batch_population_dtype,
            )
            self._f_swap_batch = torch.empty_like(self._f_batch)
        rho = torch.ones(self.res, self.res, self.res, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        if self.inlet_velocity_lu:
            ux = torch.full_like(rho, self.inlet_velocity_lu)
        feq = self.compute_equilibrium(rho, ux, uy, uz)
        self._f_batch.copy_(feq.unsqueeze(0).expand(C, -1, -1, -1, -1))
        self._f_swap_batch.copy_(self._f_batch)
        self._velocity_x_batch = ux.unsqueeze(0).expand(C, -1, -1, -1).clone()
        self._velocity_y_batch = uy.unsqueeze(0).expand(C, -1, -1, -1).clone()
        self._velocity_z_batch = uz.unsqueeze(0).expand(C, -1, -1, -1).clone()
        self._pressure_batch = (rho * (1.0 / 3.0)).unsqueeze(0).expand(C, -1, -1, -1).clone()
        self._rho_batch = rho.unsqueeze(0).expand(C, -1, -1, -1).clone()

    def _boundary_links_batch(self, mask_stack, geom_hashes=None):
        """Stacked ``[C, 26, D, H, W]`` fluid-solid links.

        Recomputes the same links ``_boundary_links`` produces per geometry but
        stacked, bypassing the single-geometry ``_boundary_link_cache``.
        """
        masks_bool = mask_stack > 0.5
        links_list = []
        for c in range(mask_stack.shape[0]):
            mask = masks_bool[c]
            fluid = ~mask
            D, H, W = mask.shape
            mask_padded = torch.nn.functional.pad(mask, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
            links = []
            for i in self._force_dirs:
                dx, dy, dz = self._stream_shifts[i]
                d_s, d_e = 1 + dx, 1 + dx + D
                h_s, h_e = 1 + dy, 1 + dy + H
                w_s, w_e = 1 + dz, 1 + dz + W
                neighbor_is_solid = mask_padded[d_s:d_e, h_s:h_e, w_s:w_e]
                links.append(fluid & neighbor_is_solid)
            links_list.append(torch.stack(links, dim=0))
        return torch.stack(links_list, dim=0)

    def _effective_drag_link_metric_exponent_batch(self, mask_stack):
        """Per-item ``[C]`` drag-link metric exponent vector.

        Same formula as ``_effective_drag_link_metric_exponent`` but computed per
        mask and never touching the shared ``self.drag_link_metric_exponent``.
        """
        exps = []
        for c in range(mask_stack.shape[0]):
            projected_cells = torch.sum(torch.any(mask_stack[c] > 0.5, dim=0).float()).item()
            projected_side = float(np.sqrt(max(projected_cells, 1.0)))
            exps.append(float(np.clip(1.68 - 0.295 * (projected_side - 13.0), 0.5, 1.68)))
        return torch.tensor(exps, device=self.device)

    def _accumulate_momentum_exchange_force_batch(self, mask_stack, boundary_links_batch=None, geom_hashes=None):
        """Vectorized 26-dir momentum-exchange sum over a batch.

        Same mask-multiply formulas as ``_accumulate_momentum_exchange_force_nosync``
        with a ``[C]`` reduction over ``(1,2,3,4)``; per-item reduction order is
        unchanged from the vectorized kernel.
        """
        if boundary_links_batch is None:
            boundary_links_batch = self._boundary_links_batch(mask_stack, geom_hashes)
        # Task 34: at force-accumulation time the ping-pong buffers are A
        # (_f_batch = pre-stream source, untouched by streaming) and B
        # (_f_swap_batch = post-stream destination). Reads the same population
        # values the 3-buffer path read from _f_pre_batch/_f_temp_batch.
        f_in_all = self._f_batch[:, 1:27].float() * boundary_links_batch
        f_out_all = self._f_swap_batch[:, self._opposite_links].float() * boundary_links_batch
        sum_all = f_in_all + f_out_all
        step_force_x = (self._force_ex_links[None, ...] * sum_all).sum(dim=(1, 2, 3, 4))
        step_force_z = (self._force_ez_links[None, ...] * sum_all).sum(dim=(1, 2, 3, 4))
        return step_force_x, step_force_z

    def _accumulate_projected_pressure_drag_proxy_batch(self, mask_stack, boundary_links_batch=None, geom_hashes=None, exponents_batch=None):
        """Batched coarse-grid pressure-drag proxy (same formula as
        ``_accumulate_projected_pressure_drag_proxy``, per-item exponent vector)."""
        if boundary_links_batch is None:
            boundary_links_batch = self._boundary_links_batch(mask_stack, geom_hashes)
        C = mask_stack.shape[0]
        flow_sign = 1.0 if self.inlet_velocity_lu >= 0.0 else -1.0
        upwind = (self._force_ex * flow_sign) > 0.0
        if exponents_batch is None:
            exponents_batch = self._effective_drag_link_metric_exponent_batch(mask_stack)
        # [1,26,1,1] link-speed metric against [C,1,1,1] exponents -> [C,26,1,1],
        # which broadcasts against the [C,26,D,H,W] projected population slice.
        metric = torch.pow(
            self._force_speed.clamp_min(1.0).view(1, 26, 1, 1),
            -exponents_batch.view(C, 1, 1, 1),
        )
        # 5-D link factors [C,26,1,1,1] so the link axis stays in dim 1 when
        # broadcast against the [C,26,D,H,W] pre-stream population slice.
        projected = (
            2.0
            * torch.abs(self._force_ex).view(1, 26, 1, 1, 1)
            * metric.unsqueeze(-1)
            * self._f_batch[:, self._force_dir_index].float()
        )
        return torch.sum(
            torch.where(upwind[None, ...], projected * boundary_links_batch, torch.zeros_like(projected)),
            dim=(1, 2, 3, 4),
        )

    def _apply_domain_boundaries_batch(self):
        """Batched inlet/outlet/mirror domain boundaries (leading batch dim).

        Operates on the post-stream destination buffer ``_f_swap_batch`` (B in
        the 2-buffer ping-pong), matching the sequential path which applies
        domain boundaries to ``self.f_temp`` after streaming.
        """
        f = self._f_swap_batch
        C = f.shape[0]
        if self.inlet_velocity_lu != 0.0:
            # Task 35: compute the inlet equilibrium in fp32 regardless of the
            # storage dtype (f may be bf16 under the knob); the write to
            # ``f`` downcasts to the storage dtype. Only the stored population
            # state differs between fp32 and bf16 runs.
            inlet_shape = f[:, :, 0, :, :].shape[2:]
            rho_in = torch.ones((C, *inlet_shape), device=self.device, dtype=torch.float32)
            ux_in = torch.full_like(rho_in, self.inlet_velocity_lu)
            uy_in = torch.zeros_like(rho_in)
            uz_in = torch.zeros_like(rho_in)
            # Batch-aware equilibrium inlet: [1,27,1,1] velocity/weight views
            # broadcast against the [C,1,H,W] macroscopic plane -> [C,27,H,W].
            cu = (
                self.ex.to(dtype=torch.float32).view(1, -1, 1, 1) * ux_in.unsqueeze(1)
                + self.ey.to(dtype=torch.float32).view(1, -1, 1, 1) * uy_in.unsqueeze(1)
                + self.ez.to(dtype=torch.float32).view(1, -1, 1, 1) * uz_in.unsqueeze(1)
            )
            u_sq = ux_in**2 + uy_in**2 + uz_in**2
            feq_in = (
                self.w.to(dtype=torch.float32).view(1, -1, 1, 1)
                * rho_in.unsqueeze(1)
                * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u_sq.unsqueeze(1))
            )
            # index_put requires matching dtypes; cast the fp32 equilibrium to
            # the storage dtype (identity for fp32).
            f[:, self._inlet_mask, 0, :, :] = feq_in[:, self._inlet_mask].to(dtype=f.dtype)

        f[:, self._outlet_mask, -1, :, :] = f[:, self._outlet_mask, -2, :, :]

        f[:, :, :, 0, :] = f[:, :, :, 1, :]
        f[:, :, :, -1, :] = f[:, :, :, -2, :]
        f[:, :, :, :, 0] = f[:, :, :, :, 1]
        f[:, :, :, :, -1] = f[:, :, :, :, -2]

    def _stream_batch_fallback(self):
        """Reference streaming fallback (27x roll): pre-stream source (buffer A,
        ``_f_batch``) -> post-stream destination (buffer B, ``_f_swap_batch``)."""
        for c in range(self._f_batch.shape[0]):
            for i in range(27):
                self._f_swap_batch[c][i] = torch.roll(
                    self._f_batch[c][i], shifts=self._stream_shifts[i], dims=(0, 1, 2)
                )

    def _apply_bfl_boundary_batch_item(self, c, mask, sparse):
        """Reference BFL for one batch item (fallback-only path) using compact q.

        Reads the pre-stream source (buffer A, ``_f_batch[c]``) and the
        plain-streamed post-stream buffer (B, ``_f_swap_batch[c]``), and
        overwrites B at the boundary-link voxels. ``sparse`` is the
        ``_build_bfl_sparse_tables`` dict; the compact per-(c, i) q slice is
        bit-identical to the full q at those voxels.
        """
        f_pre = self._f_batch[c]
        f_temp = self._f_swap_batch[c]
        boundary_links = self._boundary_links_batch(mask.unsqueeze(0))[0]
        D, H, W = mask.shape
        f_pre_padded = torch.nn.functional.pad(f_pre, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        for i in range(1, 27):
            opp_i = self._opposite_list[i]
            link_idx = i - 1
            active = boundary_links[link_idx]
            if not torch.any(active):
                continue
            pair = c * 26 + (i - 1)
            start = int(sparse["pair_start"][pair].item())
            cnt = int(sparse["pair_count"][pair].item())
            qi = sparse["q_flat"][start:start + cnt]
            active_idx = sparse["active_flat"][start:start + cnt]
            dx, dy, dz = self._stream_shifts[i]
            f_neighbor = f_pre_padded[i, 1 - dx:1 - dx + D, 1 - dy:1 - dy + H, 1 - dz:1 - dz + W].reshape(-1)[active_idx]
            f_i_here = f_pre[i].reshape(-1)[active_idx]
            f_opp_streamed = f_temp[opp_i].reshape(-1)[active_idx]
            q_low = qi < 0.5
            q_high = ~q_low
            res = torch.zeros_like(qi)
            if torch.any(q_low):
                res[q_low] = (1 - 2 * qi[q_low]) * f_neighbor[q_low] + 2 * qi[q_low] * f_i_here[q_low]
            if torch.any(q_high):
                res[q_high] = (1 / (2 * qi[q_high])) * f_i_here[q_high] + (1 - 1 / (2 * qi[q_high])) * f_opp_streamed[q_high]
            # Task 35: cast the fp32 result to the storage dtype (fp32 under
            # the default, bf16 under the knob) because index_put requires
            # matching dtypes; compute above stays fp32.
            f_temp[opp_i].reshape(-1)[active_idx] = res.to(dtype=f_temp.dtype)

    def _compute_guo_forcing_batch(self, rho, u, F, omega):
        """Batched Guo forcing source term (ext_force is unused by training)."""
        cs2 = 1.0 / 3.0
        uF = torch.sum(u * F.unsqueeze(0), dim=1)
        factor = (1.0 - 0.5 * omega)
        ei_u = torch.sum(self.ei_guo.view(1, 27, 3, 1, 1, 1) * u.unsqueeze(1), dim=2)
        ei_F = torch.sum(self.ei_guo.view(1, 27, 3, 1, 1, 1) * F.view(1, 1, 3, 1, 1, 1), dim=2)
        term1 = (ei_F - uF.unsqueeze(1)) / cs2
        term2 = (ei_u * ei_F) / (cs2**2)
        return factor * self.w.view(27, 1, 1, 1) * (term1 + term2)

    def collide_and_stream_batch(self, omega, mask_stack, ext_force=None, geom_hashes=None, q_stack=None, boundary_links_batch=None, exponents_batch=None, bfl_sparse=None):
        """One collide/stream step for C geometries at once.

        ``mask_stack`` is ``[C, D, H, W]``; ``q_stack`` (legacy) is a full
        ``[C, 27, D, H, W]`` q stack honored for backward-compatible callers.
        ``bfl_sparse`` is the compact active-voxel table from
        ``_build_bfl_sparse_tables``; when None it is built (and cached) on
        first use. Workspaces are private (``_f_batch``/``_f_swap_batch`` and
        the ``_*_batch`` accumulators); the sequential ``self.f`` state and
        single-geometry caches are never touched.
        """
        C = mask_stack.shape[0]
        device = self.device
        # Task 34 two-buffer ping-pong:
        #   A = self._f_batch      current at step start; collide in-place; then
        #                          the pre-stream source (untouched by streaming)
        #   B = self._f_swap_batch stream destination; post-stream state; every
        #                          post-stream reader uses B; then B becomes the
        #                          next step's current after the pointer swap.
        f_batch = self._f_batch
        f_swap_batch = self._f_swap_batch
        ex_b = self.ex_f.view(1, 27, 1, 1, 1)
        ey_b = self.ey_f.view(1, 27, 1, 1, 1)
        ez_b = self.ez_f.view(1, 27, 1, 1, 1)

        f_batch.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        # Task 35: fp32 compute view. When the population buffers are bf16 this
        # is a transient fp32 copy so the moment-basis matmul, the conserved
        # field reads, and the collision all run in fp32; the ONLY difference
        # between the fp32 and bf16 runs is the STORED population state.
        # Identity for fp32.
        f_calc = f_batch.float()
        has_ext_force = ext_force is not None
        N = f_batch[0, 0].numel()
        f_flat = f_calc.reshape(C, 27, N)
        # P6e FUSION-2: derive the conserved macroscopic fields (rho / jx / jy
        # / jz) from the moment-matrix rows K[:,0]/K[:,9]/K[:,3]/K[:,1] of the
        # flat-matmul collision instead of four separate full-array reductions.
        # The matmul is needed for the collision anyway; the conserved rows of
        # K_post are re-imposed from Keq below (accepted LOW-parity reduction
        # order).
        K = torch.matmul(self.moment_basis, f_flat)
        rho = K[:, 0, :].reshape(C, *f_batch.shape[2:]).clamp_min(1e-8)
        ux_raw = K[:, 9, :].reshape(C, *f_batch.shape[2:])
        uy_raw = K[:, 3, :].reshape(C, *f_batch.shape[2:])
        uz_raw = K[:, 1, :].reshape(C, *f_batch.shape[2:])
        if has_ext_force:
            ux = (ux_raw + 0.5 * ext_force[0]) / (rho + 1e-12)
            uy = (uy_raw + 0.5 * ext_force[1]) / (rho + 1e-12)
            uz = (uz_raw + 0.5 * ext_force[2]) / (rho + 1e-12)
        else:
            ux = ux_raw / (rho + 1e-12)
            uy = uy_raw / (rho + 1e-12)
            uz = uz_raw / (rho + 1e-12)
        ux = ux.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        uy = uy.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        uz = uz.nan_to_num(0.0, posinf=0.0, neginf=0.0)

        Keq = self.compute_moment_equilibrium_batch(rho, ux, uy, uz).reshape(C, 27, N)

        s_key = (float(omega), float(self.s_e), float(self.s_h))
        S = self._s_vec_cache.get(s_key)
        if S is None:
            s_vec = torch.tensor([0.0, self.s_e, float(omega), self.s_h], device=device)
            S = s_vec[self.s_indices].view(27, 1, 1, 1)
            if len(self._s_vec_cache) > 32:
                self._s_vec_cache.clear()
            self._s_vec_cache[s_key] = S

        S_flat = S.view(27, 1)
        K_post = K + S_flat * (Keq - K)

        if has_ext_force and torch.any(ext_force != 0):
            u_stacked = torch.stack([ux, uy, uz], dim=1)
            S_guo = self._compute_guo_forcing_batch(rho, u_stacked, ext_force, omega)
            K_post += torch.matmul(self.moment_basis, S_guo.reshape(C, 27, N))

        K_post[:, self.conserved_indices, :] = Keq[:, self.conserved_indices, :]

        f_new = torch.matmul(self.moment_basis_inv, K_post)
        f_batch.copy_(f_new.reshape_as(f_batch))
        f_batch.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
        # The fp32 compute view and collision transients are no longer needed;
        # free them before streaming so peak VRAM reflects only the resident
        # buffers (f_calc/f_flat are identity views in the fp32 default).
        del f_calc, f_flat, K, Keq, K_post, f_new
        # No copy into a separate pre-stream buffer: A (f_batch) remains the
        # pre-stream source after streaming below.

        used_fused_bfl = False
        if self.use_fused_stream_bfl and stream_bfl_d3q27_batch_compressed is not None:
            if bfl_sparse is None:
                bfl_sparse = self._build_bfl_sparse_tables(
                    mask_stack, geom_hashes, q_stack, boundary_links_batch
                )
            used_fused_bfl = stream_bfl_d3q27_batch_compressed(
                f_batch,
                f_swap_batch,
                bfl_sparse,
                self.ex,
                self.ey,
                self.ez,
                self.opposite,
            )

        if not used_fused_bfl:
            self._stream_batch_fallback()
            if bfl_sparse is None:
                bfl_sparse = self._build_bfl_sparse_tables(
                    mask_stack, geom_hashes, q_stack, boundary_links_batch
                )
            for c in range(C):
                self._apply_bfl_boundary_batch_item(c, mask_stack[c], bfl_sparse)

        self._apply_domain_boundaries_batch()

        step_force_x, step_force_z = self._accumulate_momentum_exchange_force_batch(
            mask_stack, boundary_links_batch=boundary_links_batch, geom_hashes=geom_hashes
        )
        step_projected_drag = self._accumulate_projected_pressure_drag_proxy_batch(
            mask_stack,
            boundary_links_batch=boundary_links_batch,
            geom_hashes=geom_hashes,
            exponents_batch=exponents_batch,
        )

        # B (f_swap_batch) is now the post-stream, domain-boundary-corrected
        # state and becomes the next step's current. nan_to_num, swap the
        # pointer roles, then recompute the macroscopic fields from the new
        # current buffer.
        f_swap_batch.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
        self._f_batch, self._f_swap_batch = self._f_swap_batch, self._f_batch
        f_batch = self._f_batch

        rho_new = torch.sum(f_batch.float(), dim=1).clamp_min(1e-8)
        if has_ext_force:
            ux_new = (torch.sum(f_batch.float() * ex_b, dim=1) + 0.5 * ext_force[0]) / (rho_new + 1e-12)
            uy_new = (torch.sum(f_batch.float() * ey_b, dim=1) + 0.5 * ext_force[1]) / (rho_new + 1e-12)
            uz_new = (torch.sum(f_batch.float() * ez_b, dim=1) + 0.5 * ext_force[2]) / (rho_new + 1e-12)
        else:
            ux_new = torch.sum(f_batch.float() * ex_b, dim=1) / (rho_new + 1e-12)
            uy_new = torch.sum(f_batch.float() * ey_b, dim=1) / (rho_new + 1e-12)
            uz_new = torch.sum(f_batch.float() * ez_b, dim=1) / (rho_new + 1e-12)
        self._velocity_x_batch = ux_new.nan_to_num(0.0)
        self._velocity_y_batch = uy_new.nan_to_num(0.0)
        self._velocity_z_batch = uz_new.nan_to_num(0.0)
        self._pressure_batch = rho_new * (1.0 / 3.0)
        self._rho_batch = rho_new

        self._force_x_last_batch = step_force_x
        self._force_z_last_batch = step_force_z
        self._projected_drag_last_batch = step_projected_drag
        if self._force_step_batch >= self._force_sample_start_batch:
            self._force_x_accum_batch += step_force_x
            self._force_z_accum_batch += step_force_z
            self._projected_drag_accum_batch += step_projected_drag
            self._force_samples_batch += 1
        self._force_step_batch += 1
        return ux, uy, uz, rho


class _DeferredAeroCoefficients:
    """Capture-side record for a deferred aerodynamic-coefficient read.

    Holds the un-read fp64 ``[15]`` GPU stack (the same 15 scalars that
    ``compute_aerodynamic_coefficients`` extracts via ``.tolist()``), the
    per-probe geometry mask, and the frozen per-solve runtime scalars the
    coefficient arithmetic depends on (``force_samples``, ``nu``,
    ``drag_link_metric_exponent``). Later solves' ``init_flow_field`` /
    ``collide_stream`` overwrite those scalars on the solver, so they are frozen
    here at capture time. ``materialize(raw_row)`` runs the identical fp64
    arithmetic from one row of the batched read (Lever 1 deferred solver reads).
    """

    __slots__ = (
        "raw_stack",
        "geometry_mask",
        "force_samples",
        "nu",
        "drag_link_metric_exponent",
        "_solver",
    )

    def __init__(
        self,
        raw_stack: torch.Tensor,
        geometry_mask: torch.Tensor,
        force_samples: int,
        nu: float,
        drag_link_metric_exponent: Optional[float],
        solver: "D3Q27CascadedSolver",
    ):
        self.raw_stack = raw_stack
        self.geometry_mask = geometry_mask
        self.force_samples = force_samples
        self.nu = nu
        self.drag_link_metric_exponent = drag_link_metric_exponent
        self._solver = solver

    def materialize(self, raw_row: Sequence[float]) -> Dict[str, float]:
        """Assemble the full coefficient dict from one row of the batched read.

        ``raw_row`` must be the 15 Python floats for this probe, in the same
        order as ``raw_stack`` (i.e. the ``.tolist()`` row of the batched
        fp64 tensor).
        """
        return self._solver._aerodynamic_coefficients_from_raw(
            self.geometry_mask,
            raw_row,
            self.force_samples,
            self.nu,
            self.drag_link_metric_exponent,
        )


class D3Q27CascadedSolver:
    """Adapter to provide a D3Q27 cascaded solver API compatible with the CFD
    simulator. Wraps the simpler `D3Q27Solver` defined above and exposes the
    same high-level methods used by `AdvancedCFDSimulator`.
    """

    def __init__(self, config, device: torch.device, phys_config):
        self.config = config
        self.device = device
        if callable(phys_config):
            self.phys_config = phys_config()
        else:
            self.phys_config = phys_config

        # resolution: config may provide `resolution` or `base_grid_resolution`
        self.resolution = getattr(self.config, 'resolution', getattr(self.config, 'base_grid_resolution', 32))

        self.inlet_velocity_lu = self._estimate_lattice_freestream_velocity()

        # instantiate the core D3Q27 solver using its expected constructor
        self._solver = D3Q27Solver(
            self.resolution,
            device,
            inlet_velocity_lu=self.inlet_velocity_lu,
            use_triton_streaming=bool(getattr(self.phys_config, "use_triton_streaming", False)),
            use_fused_stream_bfl=bool(
                getattr(self.phys_config, "use_fused_stream_bfl", False)
                if getattr(self.config, "use_fused_stream_bfl", None) is None
                else self.config.use_fused_stream_bfl
            ),
        )
        # Issue #23: Correctly pass relaxation parameters
        self._solver.s_e = float(getattr(self.phys_config, 's_e_d3q27', 1.2))
        self._solver.s_h = float(getattr(self.phys_config, 's_h_d3q27', 1.6))

        self._solver.drag_link_metric_exponent = getattr(
            self.phys_config, "drag_link_metric_exponent", self._solver.drag_link_metric_exponent
        )
        # Task 8 #2: remember whether the exponent is an explicit phys_config
        # override (non-None). In auto mode collide_stream caches the value once
        # per solve without ever clobbering an explicit override.
        self._solver._drag_exponent_is_override = (
            self._solver.drag_link_metric_exponent is not None
        )

        # Expose population arrays and buffers expected by tests and external code
        # so callers can access `solver.f` directly and observe shapes/values.
        self.f = self._solver.f
        self.f_temp = self._solver.f_temp
        self.f_pre_stream = self._solver.f_pre_stream

        # store last macroscopic fields for diagnostics (create before initialization
        # so _initialize_equilibrium can safely copy into them)
        self.velocity_x = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_y = torch.zeros_like(self.velocity_x)
        self.velocity_z = torch.zeros_like(self.velocity_x)
        self.pressure = torch.zeros_like(self.velocity_x)
        self.rho = torch.ones_like(self.velocity_x)
        self.force_x_accum = torch.tensor(0.0, device=device)
        self.force_z_accum = torch.tensor(0.0, device=device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=device)
        self.force_z_last = torch.tensor(0.0, device=device)
        self.projected_drag_accum = torch.tensor(0.0, device=device)
        self.projected_drag_last = torch.tensor(0.0, device=device)
        self.nu_turb = torch.zeros_like(self.velocity_x)
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)
        self.q_criterion = torch.zeros_like(self.velocity_x)
        self.nu = self._estimate_kinematic_viscosity()

        # Initialize populations to equilibrium immediately so tests see valid data
        # (non-NaN, correct shape) on solver construction.
        self._initialize_equilibrium()

    def _estimate_kinematic_viscosity(self):
        """Estimate lattice viscosity from lattice freestream and Reynolds."""
        Re = max(float(getattr(self.config, 'reynolds_number', 1e6)), 1e-12)
        u_lu = max(abs(float(self.inlet_velocity_lu)), 1e-6)
        # Use the lattice domain size as the reference length in lattice units.
        L_lu = float(max(self.resolution, 1))
        return max(u_lu * L_lu / Re, 1e-9)

    def _estimate_lattice_freestream_velocity(self):
        """Convert configured physical freestream to lattice units (Issue #16).

        For D3Q27, the sound speed c_s = 1/sqrt(3).
        To maintain consistent Mach number, u_lattice = Ma * c_s = Ma / sqrt(3).

        The configured Mach number is the operating-point truth: the returned
        lattice velocity is always mach / sqrt(3), never silently clamped. The
        stability envelope below is retained as a DIAGNOSTIC only -- if the
        implied lattice velocity exceeds it, a single warning is emitted per
        solver init and the unclamped value is returned unchanged.
        """
        mach = getattr(self.config, 'mach_number', 0.0)
        u_lattice = mach_to_lattice_velocity(mach)

        max_mach = float(getattr(self.phys_config, "max_mach", 0.3))
        target_lattice_velocity = float(getattr(self.phys_config, "target_lattice_velocity", 0.12))
        max_lattice_velocity = max(1e-4, min(0.85 * (max_mach / np.sqrt(3.0)), target_lattice_velocity))

        if abs(u_lattice) > max_lattice_velocity:
            logging.getLogger(__name__).warning(
                "Configured Mach %g implies lattice velocity %.6g, which exceeds the "
                "stability envelope %.6g; honoring the configured operating point "
                "unclamped (re-stabilization at this Mach is required).",
                mach,
                u_lattice,
                max_lattice_velocity,
            )

        return float(u_lattice)

    def _initialize_equilibrium(self):
        """Initialize solver populations to equilibrium with a small freestream."""
        rho = torch.ones(self.resolution, self.resolution, self.resolution, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)

        if self.inlet_velocity_lu:
            ux = torch.full_like(rho, self.inlet_velocity_lu)

        # compute and set equilibrium populations
        feq = self._solver.compute_equilibrium(rho, ux, uy, uz)
        # assign into underlying solver f (expecting shape [27, D, H, W])
        with torch.no_grad():
            self._solver.f.copy_(feq)

        # initialize stored fields
        self.velocity_x.copy_(ux)
        self.velocity_y.copy_(uy)
        self.velocity_z.copy_(uz)
        self.pressure.copy_(rho * (1.0 / 3.0))
        self.rho.copy_(rho)

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100, ext_force=None):
        """Run collide/stream for a number of steps. This adapts the simpler
        D3Q27 solver's `collide_and_stream(omega, geometry_mask)` API.
        """
        geometry_mask = geometry_mask.to(self.device, non_blocking=True)
        # compute a nominal relaxation rate from config
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)
        nu = self._estimate_kinematic_viscosity()
        self.nu = nu
        tau_min = float(getattr(self.phys_config, "tau_min_d3q27", 0.52))
        tau = max(3.0 * nu + 0.5, tau_min)
        omega = 1.0 / max(tau, 1e-12)

        sample_window = max(10, steps // 4)
        sample_start = max(0, steps - sample_window)
        self._solver.reset_force_accounting(sample_start=sample_start)

        # Issue #22: Compute hash once
        geom_hash = compute_tensor_content_hash(geometry_mask)

        # Task 8 #2: cache the drag-link metric exponent once per solve. In
        # auto mode (no explicit phys_config override) the value depends only
        # on this geometry's projected frontal area, so compute it here and let
        # the 5 steps reuse the exact same float — this removes 4 of 5 host
        # .item() syncs per solve. An explicit override is honored untouched.
        if not getattr(self._solver, "_drag_exponent_is_override", False):
            self._solver.drag_link_metric_exponent = None  # force THIS geometry
            self._solver.drag_link_metric_exponent = (
                self._solver._effective_drag_link_metric_exponent(geometry_mask)
            )

        # Issue #23: Convergence tracking
        tol = float(getattr(self.phys_config, 'convergence_tolerance', 1e-5))
        check_every = int(getattr(self.phys_config, 'check_convergence_every', 10))
        # Handle 0 case to avoid crash (Review Feedback)
        check_every = max(1, check_every)

        # run steps
        for step in range(steps):
            # v_prev feeds only the convergence check below; short training
            # solves with check_every > steps never read it, so skip the
            # 3 x N^3 stack on non-check steps (plan Phase 1).
            check_this_step = step > 0 and step % check_every == 0
            if check_this_step:
                v_prev = torch.stack([self.velocity_x, self.velocity_y, self.velocity_z])

            ux, uy, uz, rho = self._solver.collide_and_stream(
                omega, geometry_mask, ext_force=ext_force, geom_hash=geom_hash
            )

            # store fields for diagnostics BEFORE potential break (Review Feedback)
            self.velocity_x = ux
            self.velocity_y = uy
            self.velocity_z = uz
            self.pressure = rho * (1.0 / 3.0)
            self.rho = rho
            self.force_x_last = self._solver.force_x_last
            self.force_z_last = self._solver.force_z_last
            self.force_x_accum = self._solver.force_x_accum
            self.force_z_accum = self._solver.force_z_accum
            self.projected_drag_accum = self._solver.projected_drag_accum
            self.projected_drag_last = self._solver.projected_drag_last
            self.force_samples = self._solver.force_samples

            # Issue #23: Relative L2 norm convergence check
            if check_this_step:
                v_curr = torch.stack([ux, uy, uz])
                du = v_curr - v_prev
                u_mag = torch.sqrt(torch.sum(v_curr**2, dim=0) + 1e-12)
                rel_change = torch.norm(du) / (torch.norm(u_mag) + 1e-12)
                if rel_change < tol:
                    self._solver.logger.log_info(f"LBM Converged at step {step} (rel_change={rel_change:.2e} < {tol})")
                    break

    def _refresh_flow_diagnostics(self):
        """Update vorticity, Q-criterion, and turbulence proxy from the fields."""
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        gradients = compute_velocity_gradients(self.velocity_x, self.velocity_y, self.velocity_z, spacing=h)
        S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor(
            self.velocity_x, self.velocity_y, self.velocity_z, spacing=h, gradients=gradients
        )
        omega_x, omega_y, omega_z = compute_vorticity(
            self.velocity_x, self.velocity_y, self.velocity_z, spacing=h, gradients=gradients
        )

        self.vorticity[0] = omega_x
        self.vorticity[1] = omega_y
        self.vorticity[2] = omega_z

        omega_sq = omega_x**2 + omega_y**2 + omega_z**2
        strain_sq = 2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2))
        vorticity_mag = torch.sqrt(omega_sq + 1e-12)
        strain_mag = torch.sqrt(strain_sq + 1e-12)

        cs = float(getattr(self.phys_config, 'smagorinsky_constant', 0.17))
        self.nu_turb = ((cs * h) ** 2 * strain_mag).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        q_criterion = 0.5 * (omega_sq - strain_sq)
        self.q_criterion = q_criterion.nan_to_num(0.0, posinf=1e18, neginf=-1e18)

        return vorticity_mag

    def _shape_drag_correction(self, geometry_mask: torch.Tensor, projected_area_lattice: float, solid_volume: float = None):
        """Geometry-aware drag correction for non-cube voxelized bodies."""
        if not bool(getattr(self.phys_config, 'use_shape_drag_correction', True)):
            return 1.0, {}

        solid = geometry_mask > 0.5
        if solid_volume is None:
            # Fallback for callers that do not pre-extract (batched probe path).
            solid_volume = float(torch.sum(solid.float()).item())
        if solid_volume <= 0.0:
            return 1.0, {
                'shape_drag_fullness': 0.0,
                'shape_drag_blockage': 0.0,
                'shape_drag_surface_to_volume': 0.0,
            }

        x_presence = torch.any(solid, dim=(1, 2))
        x_idx = torch.where(x_presence)[0]
        # OFFLOAD-2: extract x_extent + the 3-axis surface proxy in one sync.
        has_x_extent = x_idx.numel() > 0
        extent_t = (x_idx[-1] - x_idx[0] + 1).double() if has_x_extent else solid.new_zeros(()).double()
        proxy_t0 = (solid != torch.roll(solid, shifts=1, dims=0)).float().sum().double()
        proxy_t1 = (solid != torch.roll(solid, shifts=1, dims=1)).float().sum().double()
        proxy_t2 = (solid != torch.roll(solid, shifts=1, dims=2)).float().sum().double()
        extent_val, proxy0, proxy1, proxy2 = torch.stack([extent_t, proxy_t0, proxy_t1, proxy_t2]).tolist()
        x_extent = int(extent_val) if has_x_extent else 1
        fullness = float(solid_volume / max(projected_area_lattice * max(x_extent, 1), 1.0))
        blockage = float(projected_area_lattice / max(float(self.resolution * self.resolution), 1.0))

        surface_proxy = float(proxy0 + proxy1 + proxy2)
        surface_to_volume = float(surface_proxy / max(solid_volume, 1.0))
        projected_side = float(np.sqrt(max(projected_area_lattice, 1.0)))
        log_projected_side = float(np.log(max(projected_side, 1.0)))

        # Preserve cube-like compact bodies around scale 1.0 so we do not
        # degrade already-validated baseline behavior.
        if fullness >= 0.95 and surface_to_volume <= 0.8:
            scale = 1.0
        else:
            coeffs = tuple(float(v) for v in getattr(
                self.phys_config,
                'shape_drag_correction_coefficients',
                (
                    -12.633030612111941, 27.87582461044955, -10.247055184812014,
                    22.962648171191816, -17.337224317584685, -3.946645931513679,
                    0.08323209768046214, 4.548014973469924, -5.179313884992105,
                    -7.623947231425998,
                ),
            ))
            if len(coeffs) >= 10:
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs[:10]
                b_over_f = blockage / max(fullness, 1e-6)
                log_scale = (
                    c0
                    + c1 * fullness
                    + c2 * blockage
                    + c3 * surface_to_volume
                    + c4 * (fullness * surface_to_volume)
                    + c5 * (surface_to_volume * surface_to_volume)
                    + c6 * b_over_f
                    + c7 * log_projected_side
                    + c8 * (surface_to_volume * log_projected_side)
                    + c9 * (fullness * log_projected_side)
                )
            elif len(coeffs) >= 7:
                c0, c1, c2, c3, c4, c5, c6 = coeffs[:7]
                b_over_f = blockage / max(fullness, 1e-6)
                log_scale = (
                    c0
                    + c1 * fullness
                    + c2 * blockage
                    + c3 * surface_to_volume
                    + c4 * (fullness * surface_to_volume)
                    + c5 * (surface_to_volume * surface_to_volume)
                    + c6 * b_over_f
                )
            else:
                c0, c1, c2, c3 = coeffs[:4]
                log_scale = c0 + c1 * fullness + c2 * blockage + c3 * surface_to_volume
            scale = float(np.exp(log_scale))

        min_scale = float(getattr(self.phys_config, 'shape_drag_correction_min', 0.1))
        max_scale = float(getattr(self.phys_config, 'shape_drag_correction_max', 3.0))
        scale = float(np.clip(scale, min_scale, max_scale))

        return scale, {
            'shape_drag_fullness': fullness,
            'shape_drag_blockage': blockage,
            'shape_drag_surface_to_volume': surface_to_volume,
            'shape_drag_projected_side': projected_side,
        }

    def _aerodynamic_coefficients_from_raw(
        self,
        geometry_mask: torch.Tensor,
        raw: Sequence[float],
        force_samples: int,
        nu: float,
        drag_link_metric_exponent: Optional[float],
    ) -> Dict[str, float]:
        """Run the exact fp64 arithmetic of ``compute_aerodynamic_coefficients``
        that follows the 15-scalar extraction.

        ``raw`` must be the ``.tolist()`` of the stack built below (the 15
        scalars, in the same order). The public ``compute_aerodynamic_coefficients``
        calls this with its extracted floats; the deferred path
        (``compute_aerodynamic_coefficients_deferred``) calls it later with the
        floats from the batched read. Must reproduce the full returned dict,
        INCLUDING lbm_converged, force_stability, the tiered label selection, and
        solver_quality_checks, bit-for-bit. ``force_samples``, ``nu`` and
        ``drag_link_metric_exponent`` are the frozen per-solve values captured at
        solve time (the later solves overwrite them on the solver).
        """
        (
            projected_area_raw,
            solid_volume_raw,
            projected_drag_f,
            net_drag_force_f,
            lift_force_f,
            physical_net_drag_force_f,
            physical_lift_force_f,
            force_x_accum_f,
            force_x_last_f,
            nu_turb_mean_f,
            pressure_sum_f,
            nu_turb_max_f,
            vortex_cells_f,
            vorticity_max_f,
            nan_any_f,
        ) = raw

        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        mach_number = float(getattr(self.config, 'mach_number', 0.0))

        ref_area = projected_area_raw * h**2
        ref_area = max(ref_area, h**2)
        projected_area_lattice = max(projected_area_raw, 1.0)
        raw_projected_drag_coefficient = float(projected_drag_f / projected_area_lattice)
        freestream_speed = mach_to_physical_speed(mach_number)
        drag_reference_speed = float(getattr(self.phys_config, 'drag_reference_speed', 80.0))
        speed_exponent = float(getattr(self.phys_config, 'drag_speed_normalization_exponent', 1.0))
        if freestream_speed > 1e-12 and drag_reference_speed > 0.0 and speed_exponent != 0.0:
            speed_normalization = (drag_reference_speed / freestream_speed) ** speed_exponent
        else:
            speed_normalization = 1.0

        # 2. Upwind pressure proxy (Fallback/Diagnostic)
        physical_pressure_fallback_force = raw_projected_drag_coefficient * (0.5 * 1.225 * freestream_speed**2 * ref_area)

        # 3. Tuned Surrogate version (Heuristic correction)
        shape_drag_scale, shape_drag_metrics = self._shape_drag_correction(geometry_mask, projected_area_lattice, solid_volume_raw)
        drag_coefficient_surrogate = raw_projected_drag_coefficient * speed_normalization * shape_drag_scale
        physical_surrogate_force = drag_coefficient_surrogate * (0.5 * 1.225 * freestream_speed**2 * ref_area)

        # Label Tiering Logic (Issue #12)
        # 1. lbm_raw: Pure PDE momentum exchange from internal solver
        # 2. lbm_calibrated: Heuristically corrected result for stable training

        # NOTE: the original lbm_raw_force / lbm_calibrated_force assignments
        # (formerly lines 1666-1667) are dead code -- neither feeds the returned
        # dict -- and reference GPU tensors that are not part of the deferred raw
        # vector, so they are intentionally not reproduced here.

        # Issue #16: Use pure PDE momentum exchange by default
        physical_drag_force_f = physical_net_drag_force_f
        physical_lift_force_f = physical_lift_force_f

        # Replicates _compute_force_coefficients in Python fp64 (that helper
        # calls force_x.item()/force_z.item() internally, which would re-insert
        # two syncs; inlining keeps the exact fp64 arithmetic on the extracted
        # floats).
        v_inf = mach_to_physical_speed(mach_number)
        q_inf = 0.5 * 1.225 * v_inf**2
        denom = q_inf * max(ref_area, 1e-12) + 1e-12
        drag_coefficient = float(physical_drag_force_f / denom)
        lift_coefficient = float(physical_lift_force_f / denom)
        coeffs = {
            "drag_coefficient": drag_coefficient,
            "lift_coefficient": lift_coefficient,
            "freestream_speed": v_inf,
            "density": 1.225,
        }

        # PINN-ready check: requires low divergence, stable forces, and sampling convergence
        force_stability = 1.0
        if force_samples > 20:
            avg_fx = force_x_accum_f / force_samples
            last_fx = force_x_last_f
            force_stability = abs(last_fx - avg_fx) / (abs(avg_fx) + 1e-6)

        lbm_converged = bool(
            nan_any_f == 0.0 and
            abs(force_x_last_f) < 1e5 and
            force_samples > 50 and
            force_stability < 0.1 # Relaxed for small test resolutions
        )
        compressibility_metadata = build_lbm_compressibility_metadata(
            mach_number=mach_number,
            u_lattice=self.inlet_velocity_lu,
            lbm_converged=lbm_converged,
            force_stability=force_stability,
        )

        vortex_cells = vortex_cells_f
        nu_turb_mean = nu_turb_mean_f
        reynolds_turbulent = float(v_inf * h * self.resolution / max(nu + nu_turb_mean, 1e-12))
        calibrated_drag_coefficient = float(drag_coefficient_surrogate)
        training_drag_coefficient = calibrated_drag_coefficient
        training_drag_label_source = 'lbm_calibrated'
        if lbm_converged and np.isfinite(drag_coefficient) and drag_coefficient > 0.0:
            training_drag_coefficient = drag_coefficient
            training_drag_label_source = 'lbm_raw'
        training_drag_source = str(compressibility_metadata.get('training_drag_source', 'internal_lbm_raw_low_mach'))
        if training_drag_source.startswith('none_'):
            training_drag_coefficient = None
            training_drag_label_source = training_drag_source
        lift_to_drag = float(lift_coefficient / max(abs(drag_coefficient), 1e-12))
        solver_quality_checks = {
            'finite_coefficients': bool(np.isfinite(drag_coefficient) and np.isfinite(lift_coefficient)),
            'positive_reference_area': bool(ref_area > 0.0),
            'nonempty_geometry': bool(solid_volume_raw > 0.0),
            'finite_force_outputs': bool(
                np.isfinite(physical_net_drag_force_f)
                and np.isfinite(physical_lift_force_f)
            ),
        }

        # Inline _effective_drag_link_metric_exponent (its internal .item() was
        # the last per-call sync); identical cache check + formula, with the
        # projected side read from the already-extracted area.
        drag_link_metric_exponent = (
            float(drag_link_metric_exponent)
            if drag_link_metric_exponent is not None
            else float(np.clip(1.68 - 0.295 * (float(np.sqrt(projected_area_lattice)) - 13.0), 0.5, 1.68))
        )

        force_definition = (
            'raw bounce-back momentum exchange averaged over the last-quarter window'
            if force_samples > 0
            else 'raw bounce-back momentum exchange from last streaming step'
        )

        return {
            'force_x': float(physical_drag_force_f),
            'force_z': float(physical_lift_force_f),

            # Tiered Labeling (Issue #12)
            'label_source': 'lbm_d3q27',
            'label_tier': 'lbm_raw', # Updated to raw for Issue #16
            'lbm_converged': lbm_converged,
            'force_stability': force_stability,
            **compressibility_metadata,

            'physical_force_source': float(physical_net_drag_force_f),
            'pressure_only_fallback': float(physical_pressure_fallback_force),
            'surrogate_proxy_force': float(physical_surrogate_force),

            'raw_force_x': float(projected_drag_f),
            'raw_force_z': float(lift_force_f),
            'drag_coefficient': drag_coefficient,
            'calibrated_drag_coefficient': calibrated_drag_coefficient,
            'training_drag_coefficient': training_drag_coefficient,
            'training_drag_source': training_drag_source,
            'training_drag_label_source': training_drag_label_source,
            'lift_coefficient': lift_coefficient,
            'lift_to_drag': lift_to_drag,
            'net_momentum_exchange_force_x': float(physical_net_drag_force_f),
            'raw_net_momentum_exchange_force_x': float(net_drag_force_f),
            'projected_area_lattice': projected_area_lattice,
            'raw_projected_drag_coefficient': raw_projected_drag_coefficient,
            'drag_speed_normalization': speed_normalization,
            'drag_reference_speed': drag_reference_speed,
            'drag_speed_normalization_exponent': speed_exponent,
            'shape_drag_scale': shape_drag_scale,
            'drag_link_metric_exponent': drag_link_metric_exponent,
            'force_definition': force_definition,
            'pressure_sum': float(pressure_sum_f),
            'max_turbulent_viscosity': float(nu_turb_max_f),
            'mean_smagorinsky_constant': float(getattr(self.phys_config, 'smagorinsky_constant', 0.17)),
            'max_vorticity': float(vorticity_max_f),
            'vortex_core_volume': float(vortex_cells_f * h**3),
            'reference_area': ref_area,
            'reference_area_source': 'projected_frontal_voxel_area_yz',
            'reference_area_lattice': projected_area_lattice,
            'reference_length': h * self.resolution,
            'reference_length_source': 'grid_spacing_times_resolution',
            'freestream_speed': v_inf,
            'density': coeffs['density'],
            'reynolds_number_turbulent': reynolds_turbulent,
            'empty_geometry': bool(solid_volume_raw <= 0.0),
            'claim_bearing_cfd': False,
            'solver_quality_checks': solver_quality_checks,
            'solver_provenance': {
                'primary_solver': 'D3Q27',
                'label_tier': 'lbm_raw',
                'lbm_converged': lbm_converged,
                'grid_resolution': int(self.resolution),
                'force_samples': int(force_samples),
                'reference_area_source': 'projected_frontal_voxel_area_yz',
            },
        } | shape_drag_metrics

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        """Compute approximate aerodynamic coefficients from the last simulated
        macroscopic fields. This mirrors the interface used by the training
        CFD simulator.
        """
        # conservative reference area and freestream speed
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        solid = geometry_mask > 0.5

        # OFFLOAD-2: collapse the ~25 per-solve GPU->CPU .item() reads into a
        # single stacked .tolist() (one sync). Every reduction is computed on
        # GPU and widened fp32->fp64 on GPU; fp64 widening is lossless, so each
        # unpacked Python float is bit-identical to float(x.item()). All
        # downstream coefficient arithmetic stays in Python fp64, unchanged.
        mach_number = float(getattr(self.config, 'mach_number', 0.0))
        force_samples = self._solver.force_samples

        # Issue #23: Handle zero-sample case for early convergence (Review Feedback)
        if force_samples > 0:
            projected_drag = self._solver.projected_drag_accum / force_samples
            net_drag_force = self._solver.force_x_accum / force_samples
            lift_force = self._solver.force_z_accum / force_samples
        else:
            projected_drag = self._solver.projected_drag_last
            net_drag_force = self._solver.force_x_last
            lift_force = self._solver.force_z_last

        # 1. Pure momentum exchange (Raw PDE Ground Truth)
        physical_net_drag_force = _scale_momentum_exchange_force(net_drag_force, h, mach_number)
        physical_lift_force = _scale_momentum_exchange_force(lift_force, h, mach_number)

        vorticity_mag = self._refresh_flow_diagnostics()

        q_threshold = float(getattr(self.phys_config, 'q_threshold', 0.0))
        raw_scalars = torch.stack([
            torch.any(solid, dim=0).float().sum().double(),
            solid.float().sum().double(),
            projected_drag.double(),
            net_drag_force.double(),
            lift_force.double(),
            physical_net_drag_force.double(),
            physical_lift_force.double(),
            self._solver.force_x_accum.double(),
            self._solver.force_x_last.double(),
            self.nu_turb.mean().double(),
            self.pressure.sum().double(),
            self.nu_turb.max().double(),
            (self.q_criterion > q_threshold).float().sum().double(),
            vorticity_mag.max().double(),
            torch.isnan(self.velocity_x).any().double(),
        ]).tolist()
        return self._aerodynamic_coefficients_from_raw(
            geometry_mask,
            raw_scalars,
            force_samples,
            self.nu,
            self._solver.drag_link_metric_exponent,
        )

    def compute_aerodynamic_coefficients_deferred(
        self, geometry_mask: torch.Tensor
    ) -> "_DeferredAeroCoefficients":
        """Compute the same 15-scalar stack as ``compute_aerodynamic_coefficients``
        but return it UN-READ (fp64 GPU ``[15]`` tensor) plus the frozen
        per-solve runtime scalars (``force_samples``, ``nu``,
        ``drag_link_metric_exponent``). The SPSA probe loop can then enqueue all
        solves back-to-back with NO host scalar reads and read every probe's
        scalars in ONE batched ``torch.stack(...).tolist()`` afterwards
        (Lever 1 deferred solver reads). ``materialize()`` applies the identical
        fp64 arithmetic per probe from the batched row.

        The frozen scalars are captured here (immediately after this solve's
        ``collide_stream``); later solves' ``init_flow_field``/``collide_stream``
        overwrite them on the solver, so the deferred materialize must not read
        them back from ``self``.
        """
        # conservative reference area and freestream speed
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        solid = geometry_mask > 0.5

        # Same branch selection + 15-scalar stack as
        # compute_aerodynamic_coefficients; identical fp32->fp64 widenings on
        # GPU, but no .tolist() here.
        mach_number = float(getattr(self.config, 'mach_number', 0.0))
        force_samples = self._solver.force_samples

        # Issue #23: Handle zero-sample case for early convergence (Review Feedback)
        if force_samples > 0:
            projected_drag = self._solver.projected_drag_accum / force_samples
            net_drag_force = self._solver.force_x_accum / force_samples
            lift_force = self._solver.force_z_accum / force_samples
        else:
            projected_drag = self._solver.projected_drag_last
            net_drag_force = self._solver.force_x_last
            lift_force = self._solver.force_z_last

        # 1. Pure momentum exchange (Raw PDE Ground Truth)
        physical_net_drag_force = _scale_momentum_exchange_force(net_drag_force, h, mach_number)
        physical_lift_force = _scale_momentum_exchange_force(lift_force, h, mach_number)

        vorticity_mag = self._refresh_flow_diagnostics()

        q_threshold = float(getattr(self.phys_config, 'q_threshold', 0.0))
        raw_stack = torch.stack([
            torch.any(solid, dim=0).float().sum().double(),
            solid.float().sum().double(),
            projected_drag.double(),
            net_drag_force.double(),
            lift_force.double(),
            physical_net_drag_force.double(),
            physical_lift_force.double(),
            self._solver.force_x_accum.double(),
            self._solver.force_x_last.double(),
            self.nu_turb.mean().double(),
            self.pressure.sum().double(),
            self.nu_turb.max().double(),
            (self.q_criterion > q_threshold).float().sum().double(),
            vorticity_mag.max().double(),
            torch.isnan(self.velocity_x).any().double(),
        ])
        return _DeferredAeroCoefficients(
            raw_stack=raw_stack,
            geometry_mask=geometry_mask,
            force_samples=force_samples,
            nu=self.nu,
            drag_link_metric_exponent=self._solver.drag_link_metric_exponent,
            solver=self,
        )

    # ------------------------------------------------------------------
    # Task 10: batched SPSA probe path (private workspaces on the inner
    # solver). The sequential collide_stream / compute_aerodynamic_coefficients
    # remain authoritative and are not called from these methods.
    # ------------------------------------------------------------------
    def collide_stream_batch(self, geometry_masks, steps: int = 100, geom_hashes=None, q_stack=None, ext_force=None):
        """Run collide/stream for a batch of C geometries at once.

        ``geometry_masks`` is a ``[C, D, H, W]`` binary float tensor. Returns a
        list of C per-item coefficient dicts (at least ``drag_coefficient`` and
        ``lift_coefficient``), each reproducing the schema a single sequential
        solve produces.
        """
        C = geometry_masks.shape[0]
        geometry_masks = geometry_masks.to(self.device, non_blocking=True)
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)
        nu = self._estimate_kinematic_viscosity()
        self.nu = nu
        tau_min = float(getattr(self.phys_config, "tau_min_d3q27", 0.52))
        tau = max(3.0 * nu + 0.5, tau_min)
        omega = 1.0 / max(tau, 1e-12)

        sample_window = max(10, steps // 4)
        sample_start = max(0, steps - sample_window)
        self._solver.reset_force_accounting_batch(C, sample_start=sample_start)

        if geom_hashes is None:
            geom_hashes = [compute_tensor_content_hash(geometry_masks[c]) for c in range(C)]

        # Geometry-only precompute, reused across all steps: stacked boundary
        # links and the per-item drag-link metric exponent. Never populates the
        # single-geometry _boundary_link_cache or drag_link_metric_exponent.
        boundary_links_batch = self._solver._boundary_links_batch(geometry_masks, geom_hashes)
        exponents_batch = self._solver._effective_drag_link_metric_exponent_batch(geometry_masks)

        # Task 34: build the compact active-voxel q tables once (geometry-static)
        # instead of materializing a [C, 27, D, H, W] q stack on GPU. If the
        # caller supplied a q_stack it is honored; otherwise per-geometry q is
        # computed transiently and only boundary q survives on GPU.
        bfl_sparse = self._solver._build_bfl_sparse_tables(
            geometry_masks, geom_hashes, q_stack, boundary_links_batch
        )

        self._solver._init_batch_equilibrium(C)

        for step in range(steps):
            ux, uy, uz, rho = self._solver.collide_and_stream_batch(
                omega,
                geometry_masks,
                ext_force=ext_force,
                geom_hashes=geom_hashes,
                q_stack=q_stack,
                boundary_links_batch=boundary_links_batch,
                exponents_batch=exponents_batch,
                bfl_sparse=bfl_sparse,
            )
            # Mirror the single-solve field aliasing for diagnostics (item 0).
            self.velocity_x = ux[0]
            self.velocity_y = uy[0]
            self.velocity_z = uz[0]
            self.pressure = rho[0] * (1.0 / 3.0)
            self.rho = rho[0]

        return self.compute_aerodynamic_coefficients_batch(geometry_masks)

    def compute_aerodynamic_coefficients_batch(self, geometry_masks):
        """Reproduce ``compute_aerodynamic_coefficients`` per item from the
        batched force accounting and velocity fields. Returns a list of C dicts.
        """
        C = geometry_masks.shape[0]
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        solver = self._solver
        results = []
        for c in range(C):
            mask = geometry_masks[c]
            solid = mask > 0.5
            ref_area = torch.sum(torch.any(solid, dim=0).float()).item() * h**2
            ref_area = max(ref_area, h**2)

            force_samples = int(solver._force_samples_batch[c].item())
            if force_samples > 0:
                projected_drag = solver._projected_drag_accum_batch[c] / force_samples
                net_drag_force = solver._force_x_accum_batch[c] / force_samples
                lift_force = solver._force_z_accum_batch[c] / force_samples
                force_definition = 'raw bounce-back momentum exchange averaged over the last-quarter window'
            else:
                projected_drag = solver._projected_drag_last_batch[c]
                net_drag_force = solver._force_x_last_batch[c]
                lift_force = solver._force_z_last_batch[c]
                force_definition = 'raw bounce-back momentum exchange from last streaming step'

            projected_area_lattice = max(torch.sum(torch.any(solid, dim=0).float()).item(), 1.0)
            raw_projected_drag_coefficient = float(projected_drag.item() / projected_area_lattice)
            freestream_speed = mach_to_physical_speed(float(getattr(self.config, 'mach_number', 0.0)))
            drag_reference_speed = float(getattr(self.phys_config, 'drag_reference_speed', 80.0))
            speed_exponent = float(getattr(self.phys_config, 'drag_speed_normalization_exponent', 1.0))
            if freestream_speed > 1e-12 and drag_reference_speed > 0.0 and speed_exponent != 0.0:
                speed_normalization = (drag_reference_speed / freestream_speed) ** speed_exponent
            else:
                speed_normalization = 1.0
            physical_net_drag_force = _scale_momentum_exchange_force(
                net_drag_force, h, getattr(self.config, 'mach_number', 0.0)
            )
            physical_lift_force = _scale_momentum_exchange_force(
                lift_force, h, getattr(self.config, 'mach_number', 0.0)
            )

            physical_pressure_fallback_force = raw_projected_drag_coefficient * (
                0.5 * 1.225 * freestream_speed**2 * ref_area
            )

            shape_drag_scale, shape_drag_metrics = self._shape_drag_correction(mask, projected_area_lattice)
            drag_coefficient_surrogate = raw_projected_drag_coefficient * speed_normalization * shape_drag_scale
            physical_surrogate_force = drag_coefficient_surrogate * (0.5 * 1.225 * freestream_speed**2 * ref_area)

            lbm_raw_force = physical_net_drag_force
            lbm_calibrated_force = torch.tensor(physical_surrogate_force, device=self.device, dtype=self.f.dtype)
            physical_drag_force = lbm_raw_force

            coeffs = _compute_force_coefficients(
                physical_drag_force,
                physical_lift_force,
                getattr(self.config, 'mach_number', 0.0),
                ref_area=max(ref_area, 1e-12),
                rho_ref=1.225,
            )

            force_stability = 1.0
            if force_samples > 20:
                avg_fx = float(solver._force_x_accum_batch[c].item()) / force_samples
                last_fx = float(solver._force_x_last_batch[c].item())
                force_stability = abs(last_fx - avg_fx) / (abs(avg_fx) + 1e-6)

            lbm_converged = bool(
                not torch.isnan(solver._velocity_x_batch[c]).any()
                and abs(float(solver._force_x_last_batch[c].item())) < 1e5
                and force_samples > 50
                and force_stability < 0.1
            )
            compressibility_metadata = build_lbm_compressibility_metadata(
                mach_number=getattr(self.config, 'mach_number', 0.0),
                u_lattice=self.inlet_velocity_lu,
                lbm_converged=lbm_converged,
                force_stability=force_stability,
            )

            drag_coefficient = float(coeffs['drag_coefficient'])
            lift_coefficient = float(coeffs['lift_coefficient'])
            calibrated_drag_coefficient = float(drag_coefficient_surrogate)
            training_drag_coefficient = calibrated_drag_coefficient
            training_drag_label_source = 'lbm_calibrated'
            if lbm_converged and np.isfinite(drag_coefficient) and drag_coefficient > 0.0:
                training_drag_coefficient = drag_coefficient
                training_drag_label_source = 'lbm_raw'
            training_drag_source = str(compressibility_metadata.get('training_drag_source', 'internal_lbm_raw_low_mach'))
            if training_drag_source.startswith('none_'):
                training_drag_coefficient = None
                training_drag_label_source = training_drag_source
            lift_to_drag = float(lift_coefficient / max(abs(drag_coefficient), 1e-12))

            v_inf = coeffs.get('freestream_speed', 0.0)

            # reynolds_number_turbulent is explicit-unavailable (None) on the
            # batch path: it needs nu_turb_mean, the mean Smagorinsky
            # turbulent-viscosity field, which is computed only in the SEQUENTIAL
            # compute_aerodynamic_coefficients (:1765). A hardcoded
            # nu_turb_mean = 0.0 here would silently report a LAMINAR-ONLY
            # Reynolds number under a "turbulent" key (a mislabel). The consumer
            # audit found nothing reads reynolds_number_turbulent from batch
            # dicts, so the unavailable marking is propagated instead, consistent
            # with the three turbulence/vorticity fields below.

            results.append({
                'force_x': float(physical_drag_force.item() if isinstance(physical_drag_force, torch.Tensor) else physical_drag_force),
                'force_z': float(physical_lift_force.item() if isinstance(physical_lift_force, torch.Tensor) else physical_lift_force),
                'label_source': 'lbm_d3q27',
                'label_tier': 'lbm_raw',
                'lbm_converged': lbm_converged,
                'force_stability': force_stability,
                **compressibility_metadata,
                'physical_force_source': float(physical_net_drag_force.item() if isinstance(physical_net_drag_force, torch.Tensor) else physical_net_drag_force),
                'pressure_only_fallback': float(physical_pressure_fallback_force),
                'surrogate_proxy_force': float(physical_surrogate_force),
                'raw_force_x': float(projected_drag.item() if isinstance(projected_drag, torch.Tensor) else projected_drag),
                'raw_force_z': float(lift_force.item() if isinstance(lift_force, torch.Tensor) else lift_force),
                'drag_coefficient': drag_coefficient,
                'calibrated_drag_coefficient': calibrated_drag_coefficient,
                'training_drag_coefficient': training_drag_coefficient,
                'training_drag_source': training_drag_source,
                'training_drag_label_source': training_drag_label_source,
                'lift_coefficient': lift_coefficient,
                'lift_to_drag': lift_to_drag,
                'net_momentum_exchange_force_x': float(physical_net_drag_force.item() if isinstance(physical_net_drag_force, torch.Tensor) else physical_net_drag_force),
                'raw_net_momentum_exchange_force_x': float(net_drag_force.item() if isinstance(net_drag_force, torch.Tensor) else net_drag_force),
                'projected_area_lattice': projected_area_lattice,
                'raw_projected_drag_coefficient': raw_projected_drag_coefficient,
                'drag_speed_normalization': speed_normalization,
                'drag_reference_speed': drag_reference_speed,
                'drag_speed_normalization_exponent': speed_exponent,
                'shape_drag_scale': shape_drag_scale,
                'force_definition': force_definition,
                'pressure_sum': float(solver._pressure_batch[c].sum().item()),
                # Explicit-unavailable on the batch path: the batched force
                # accounting holds force sums and the velocity fields but NOT the
                # per-geometry vorticity / Smagorinsky turbulent-viscosity
                # fields. Those are computed only in the SEQUENTIAL
                # compute_aerodynamic_coefficients (:1838 max_turbulent_viscosity
                # from nu_turb_max_f, :1840 max_vorticity from vorticity_max_f,
                # :1841 vortex_core_volume from vortex_cells_f). Computing them
                # here would require per-geometry curl + viscosity reductions over
                # the full field with no existing parity gate. None serializes to
                # JSON null = unambiguous "not computed here".
                'max_turbulent_viscosity': None,
                'mean_smagorinsky_constant': float(getattr(self.phys_config, 'smagorinsky_constant', 0.17)),
                'max_vorticity': None,
                'vortex_core_volume': None,
                'reference_area': ref_area,
                'reference_area_source': 'projected_frontal_voxel_area_yz',
                'reference_area_lattice': projected_area_lattice,
                'reference_length': h * self.resolution,
                'reference_length_source': 'grid_spacing_times_resolution',
                'freestream_speed': v_inf,
                'density': coeffs['density'],
                'reynolds_number_turbulent': None,
                'empty_geometry': bool(torch.sum(solid.float()).item() <= 0.0),
                'claim_bearing_cfd': False,
                'solver_quality_checks': {
                    'finite_coefficients': bool(np.isfinite(drag_coefficient) and np.isfinite(lift_coefficient)),
                    'positive_reference_area': bool(ref_area > 0.0),
                    'nonempty_geometry': bool(torch.sum(solid.float()).item() > 0.0),
                    'finite_force_outputs': bool(
                        np.isfinite(float(physical_net_drag_force.item() if isinstance(physical_net_drag_force, torch.Tensor) else physical_net_drag_force))
                        and np.isfinite(float(physical_lift_force.item() if isinstance(physical_lift_force, torch.Tensor) else physical_lift_force))
                    ),
                },
                'solver_provenance': {
                    'primary_solver': 'D3Q27',
                    'label_tier': 'lbm_raw',
                    'lbm_converged': lbm_converged,
                    'grid_resolution': int(self.resolution),
                    'force_samples': int(force_samples),
                    'reference_area_source': 'projected_frontal_voxel_area_yz',
                },
            } | shape_drag_metrics)
        return results


if __name__ == '__main__':
    import argparse
    import os
    import time
    try:
        import trimesh
    except Exception:
        trimesh = None
    from scipy.ndimage import zoom

    parser = argparse.ArgumentParser(description='Run the D3Q27 LBM solver on an input STL')
    parser.add_argument('stl', help='Path to input STL file')
    parser.add_argument('--grid', type=int, default=64, help='Solver grid resolution (default: 64)')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps (default: 500)')
    parser.add_argument('--mach', type=float, default=0.025, help='Mach number (default: 0.025)')
    parser.add_argument('--re', type=float, default=1e5, help='Reynolds number (default: 1e5)')
    parser.add_argument('--body-size', type=float, default=1.0, help='Physical body size in meters (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Device to run on')
    args = parser.parse_args()

    if not os.path.exists(args.stl):
        raise FileNotFoundError(f"STL file not found: {args.stl}")

    if trimesh is None:
        raise ImportError('trimesh is required to voxelize STL files; please install trimesh')

    # Load mesh
    mesh = trimesh.load_mesh(args.stl)

    # Domain and grid spacing
    grid_resolution = int(args.grid)
    body_size = float(args.body_size)
    # Keep a fluid buffer around the body so the mesh does not fill the full box.
    domain_scale = 2.0
    domain_size = [body_size * domain_scale, body_size * domain_scale, body_size * domain_scale]
    grid_spacing = domain_size[0] / float(grid_resolution)

    # Preserve mesh physical size and center in domain
    bounds = mesh.bounds
    mesh_extent = bounds[1] - bounds[0]
    max_mesh_extent = float(np.max(mesh_extent)) if mesh_extent is not None else 1.0
    if max_mesh_extent > 1e-12:
        scale_factor = body_size / max_mesh_extent
        mesh.vertices = (mesh.vertices - bounds[0]) * scale_factor

    mesh_center = np.mean(mesh.vertices, axis=0)
    domain_center = np.array(domain_size) / 2.0
    mesh.vertices = mesh.vertices - mesh_center + domain_center

    # Voxelize using trimesh voxelization
    voxel_pitch = grid_spacing
    try:
        voxel_grid = mesh.voxelized(voxel_pitch).fill()
        voxel_np = voxel_grid.matrix.view(np.ndarray)
    except Exception:
        # fall back to coarser pitch
        voxel_pitch = voxel_pitch * 2.0
        voxel_grid = mesh.voxelized(voxel_pitch).fill()
        voxel_np = voxel_grid.matrix.view(np.ndarray)

    # Resize to requested grid by placing occupied voxel centers into the domain grid.
    # This preserves the body size and leaves surrounding fluid cells for the wake.
    target_shape = (grid_resolution, grid_resolution, grid_resolution)
    resized = np.zeros(target_shape, dtype=np.float32)
    try:
        voxel_points = voxel_grid.points
        voxel_indices = np.rint(voxel_points / voxel_pitch).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid_resolution - 1)
        resized[
            voxel_indices[:, 0],
            voxel_indices[:, 1],
            voxel_indices[:, 2],
        ] = 1.0
    except Exception:
        # Fallback: preserve the old behavior if point-based voxel placement fails.
        voxel_np = voxel_grid.matrix.view(np.ndarray)
        zoom_factors = np.array(target_shape) / np.array(voxel_np.shape)
        resized = zoom(voxel_np.astype(np.float32), zoom_factors, order=1)
        resized = (resized > 0.5).astype(np.float32)

    # Convert to torch tensor and device
    import torch
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    geometry_mask = torch.from_numpy(resized).float().to(device)

    # Minimal config objects expected by D3Q27CascadedSolver
    class _Cfg:
        def __init__(self, resolution, mach, reynolds):
            self.resolution = resolution
            self.mach_number = mach
            self.reynolds_number = reynolds
            self.lbm_config = type('LC', (), {})()

    # Minimal phys config
    class _Phys:
        def __init__(self):
            self.smagorinsky_constant = 0.17
            self.test_filter_ratio = 2.0
            self.dynamic_cs_clip_min = 0.0
            self.dynamic_cs_clip_max = 0.2
            self.wale_constant = 0.5
            self.use_les_turbulence = True
            self.turbulence_model = 'dynamic_smagorinsky'
            self.use_vorticity_confinement = True
            self.vc_adaptive = True
            self.vorticity_confinement_epsilon = 0.1
            self.compute_q_criterion = True
            self.check_convergence_every = 250
            self.convergence_tolerance = 1e-5
            self.q_threshold = 0.0
            self.s_nu = None
            self.s_bulk = 1.0
            self.s_energy = 1.2
            self.s_higher = 1.4
            self.tau_min_d3q27 = 0.52
            self.use_triton_streaming = False
            self.drag_link_metric_exponent = None
            self.drag_reference_speed = 80.0
            self.drag_speed_normalization_exponent = 1.0
            self.use_shape_drag_correction = True
            self.shape_drag_correction_coefficients = (
                -12.633030612111941, 27.87582461044955, -10.247055184812014,
                22.962648171191816, -17.337224317584685, -3.946645931513679,
                0.08323209768046214, 4.548014973469924, -5.179313884992105,
                -7.623947231425998,
            )
            self.shape_drag_correction_min = 0.1
            self.shape_drag_correction_max = 3.0
            self.target_lattice_velocity = 0.12
            self.max_mach = 0.3

    # Populate lbm_config fields commonly used
    cfg = _Cfg(grid_resolution, args.mach, args.re)
    cfg.lbm_config.grid_spacing = grid_spacing
    cfg.lbm_config.time_step = 1e-3
    cfg.lbm_config.physical_length_scale = body_size
    cfg.lbm_config.compute_q_criterion = True
    cfg.lbm_config.use_vorticity_confinement = True

    phys = _Phys()

    # Create solver and run
    print(f"Running solver on {args.stl} with grid={grid_resolution}, steps={args.steps}, device={device}")
    solver = D3Q27CascadedSolver(cfg, device, phys)

    t0 = time.time()
    solver.collide_stream(geometry_mask, steps=int(args.steps))
    t1 = time.time()

    # Compute and print aerodynamic coefficients
    try:
        coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)
        print('Simulation complete:')
        print(f"  Force X: {coeffs['force_x']:.6e}")
        print(f"  Force Z: {coeffs['force_z']:.6e}")
        print(f"  Coefficient of Drag (Cd): {coeffs['drag_coefficient']:.6e}")
        print(f"  Coefficient of Lift (Cl): {coeffs['lift_coefficient']:.6e}")
        print(f"  Force Definition: {coeffs['force_definition']}")
        print(f"  Pressure Sum: {coeffs['pressure_sum']:.6e}")
        print(f"  Max Turbulent Viscosity: {coeffs['max_turbulent_viscosity']:.6e}")
        print(f"  Mean Smagorinsky Constant: {coeffs['mean_smagorinsky_constant']:.6e}")
        print(f"  Max Vorticity: {coeffs['max_vorticity']:.6e}")
        print(f"  Vortex Core Volume: {coeffs['vortex_core_volume']:.6e}")
        print(f"  Reference Area: {coeffs['reference_area']:.6e}")
        print(f"  Reference Length: {coeffs['reference_length']:.6e}")
        print(f"  Freestream Speed: {coeffs['freestream_speed']:.6e}")
        print(f"  Density: {coeffs['density']:.6e}")
        print(f"  Reynolds Number (turbulent): {coeffs['reynolds_number_turbulent']:.6e}")
    except Exception as e:
        print('Could not compute aerodynamic coefficients:', e)

    print(f"Elapsed time: {t1 - t0:.2f}s")
