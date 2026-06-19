import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from advanced_lbm_solver import D3Q27CascadedSolver
from lbm_utils import mach_to_lattice_velocity
from thermal_lbm_solver import (
    ThermalBGKSolver,
    ThermalD3Q27Solver,
    ThermalLBMConfig,
    create_thermal_lbm_solver,
)


class TestPhysicsConfig:
    max_mach = 2.0
    target_lattice_velocity = 0.2
    tau_min_d3q27 = 0.52
    s_e_d3q27 = 1.2
    s_h_d3q27 = 1.6
    drag_link_metric_exponent = None
    use_triton_streaming = False
    convergence_tolerance = 1e-5
    check_convergence_every = 10
    smagorinsky_constant = 0.17
    q_threshold = 0.0
    use_shape_drag_correction = False


def make_config(mach_number: float = 0.2, resolution: int = 8, thermal_enabled: bool = True):
    lbm_config = SimpleNamespace(grid_spacing=0.125, time_step=0.001, physical_length_scale=1.0)
    return SimpleNamespace(
        base_grid_resolution=resolution,
        resolution=resolution,
        mach_number=mach_number,
        reynolds_number=100.0,
        simulation_steps=1,
        lbm_config=lbm_config,
        thermal_enabled=thermal_enabled,
        thermal_model="coupled_d3q7_temperature_bgk" if thermal_enabled else "none_isothermal",
        thermal_lbm_config=ThermalLBMConfig(
            reference_temperature=310.0,
            thermal_diffusivity_lattice=0.015,
            max_thermal_steps_per_call=4,
        ),
    )


class TestThermalLBMSolver(unittest.TestCase):
    def test_thermal_bgk_tensors_stay_on_requested_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        solver = ThermalBGKSolver((4, 4, 4), device, ThermalLBMConfig())

        self.assertEqual(solver.g.device.type, device.type)
        self.assertEqual(solver.temperature.device.type, device.type)
        self.assertEqual(solver.weights.device.type, device.type)

    def test_uniform_temperature_is_stationary_without_flow(self):
        device = torch.device("cpu")
        solver = ThermalBGKSolver((6, 6, 6), device, ThermalLBMConfig(reference_temperature=321.0))
        geometry = torch.zeros((6, 6, 6), dtype=torch.float32, device=device)
        zero_velocity = tuple(torch.zeros_like(geometry) for _ in range(3))

        solver.collide_stream(zero_velocity, geometry, steps=3)

        self.assertLess(float(torch.max(torch.abs(solver.temperature - 321.0)).item()), 1e-5)

    def test_thermal_geometry_uses_flow_solver_solid_threshold(self):
        device = torch.device("cpu")
        config = ThermalLBMConfig(reference_temperature=300.0)
        fluid_solver = ThermalBGKSolver((5, 5, 5), device, config)
        threshold_solver = ThermalBGKSolver((5, 5, 5), device, config)
        temperature = torch.full((5, 5, 5), 300.0, dtype=torch.float32, device=device)
        temperature[2, 2, 2] = 340.0
        zero_velocity = tuple(torch.zeros_like(temperature) for _ in range(3))
        fluid_solver.set_temperature(temperature)
        threshold_solver.set_temperature(temperature)

        fluid_geometry = torch.zeros_like(temperature)
        below_threshold_geometry = torch.zeros_like(temperature)
        below_threshold_geometry[2, 2, 2] = 0.25

        fluid_solver.collide_stream(zero_velocity, fluid_geometry, steps=1)
        threshold_solver.collide_stream(zero_velocity, below_threshold_geometry, steps=1)

        self.assertTrue(torch.allclose(threshold_solver.temperature, fluid_solver.temperature))

    def test_solid_cells_do_not_emit_thermal_populations_to_fluid(self):
        device = torch.device("cpu")
        solver = ThermalBGKSolver((5, 5, 5), device, ThermalLBMConfig(reference_temperature=300.0))
        temperature = torch.full((5, 5, 5), 300.0, dtype=torch.float32, device=device)
        temperature[2, 2, 2] = 1000.0
        geometry = torch.zeros_like(temperature)
        geometry[2, 2, 2] = 1.0
        zero_velocity = tuple(torch.zeros_like(temperature) for _ in range(3))
        solver.set_temperature(temperature)

        solver.collide_stream(zero_velocity, geometry, steps=1)

        adjacent = torch.tensor(
            [
                solver.temperature[1, 2, 2],
                solver.temperature[3, 2, 2],
                solver.temperature[2, 1, 2],
                solver.temperature[2, 3, 2],
                solver.temperature[2, 2, 1],
                solver.temperature[2, 2, 3],
            ],
            device=device,
        )
        self.assertTrue(torch.allclose(adjacent, torch.full_like(adjacent, 300.0)))

    def test_shock_sensor_is_low_for_uniform_temperature_and_high_at_jump(self):
        device = torch.device("cpu")
        solver = ThermalBGKSolver((8, 4, 4), device, ThermalLBMConfig(shock_sensor_threshold=0.02))
        uniform = torch.full((8, 4, 4), 300.0, dtype=torch.float32, device=device)
        jump = uniform.clone()
        jump[4:, :, :] = 600.0

        uniform_sensor = solver.compute_shock_sensor(uniform)
        jump_sensor = solver.compute_shock_sensor(jump)

        self.assertLess(float(uniform_sensor.max().item()), 1e-6)
        self.assertGreater(float(jump_sensor.max().item()), 0.5)

    def test_shock_stabilization_lowers_local_omega_near_jump(self):
        device = torch.device("cpu")
        config = ThermalLBMConfig(
            reference_temperature=300.0,
            shock_stabilization_enabled=True,
            shock_diffusivity_multiplier=4.0,
        )
        solver = ThermalBGKSolver((8, 4, 4), device, config)
        temperature = torch.full((8, 4, 4), 300.0, dtype=torch.float32, device=device)
        temperature[4:, :, :] = 600.0

        sensor = solver.compute_shock_sensor(temperature)
        omega_field = solver.compute_effective_omega(sensor)

        self.assertLess(float(omega_field[sensor > 0.5].mean().item()), float(solver.omega))

    def test_thermal_inlet_outlet_boundaries_set_inlet_and_extrapolate_outlet(self):
        device = torch.device("cpu")
        config = ThermalLBMConfig(
            reference_temperature=300.0,
            inlet_temperature=350.0,
            outlet_temperature=None,
            thermal_boundary_model="fixed_temperature_inlet_zero_gradient_outlet",
        )
        solver = ThermalBGKSolver((6, 4, 4), device, config)
        geometry = torch.zeros((6, 4, 4), dtype=torch.float32, device=device)
        zero_velocity = tuple(torch.zeros_like(geometry) for _ in range(3))
        solver.temperature[-2, :, :] = 325.0
        solver.g.copy_(solver.compute_equilibrium(solver.temperature, zero_velocity))

        solver.collide_stream(zero_velocity, geometry, steps=1)

        self.assertTrue(torch.allclose(solver.temperature[0], torch.full_like(solver.temperature[0], 350.0)))
        self.assertTrue(torch.allclose(solver.temperature[-1], solver.temperature[-2]))

    def test_pressure_density_temperature_consistency(self):
        device = torch.device("cpu")
        config = ThermalLBMConfig(gas_constant=287.05)
        solver = ThermalBGKSolver((4, 4, 4), device, config)
        rho = torch.full((4, 4, 4), 1.2, device=device)
        temperature = torch.full((4, 4, 4), 288.15, device=device)

        state = solver.build_thermodynamic_state(rho, temperature)

        self.assertTrue(torch.allclose(state.pressure, rho * config.gas_constant * temperature))
        self.assertTrue(torch.allclose(state.density, rho))
        self.assertTrue(torch.allclose(state.temperature, temperature))

    def test_positivity_guards_clamp_invalid_state(self):
        device = torch.device("cpu")
        config = ThermalLBMConfig(min_temperature=50.0, min_density=0.01, min_pressure=1.0)
        solver = ThermalBGKSolver((2, 2, 2), device, config)
        rho = torch.full((2, 2, 2), -2.0, device=device)
        temperature = torch.full((2, 2, 2), -100.0, device=device)

        state = solver.build_thermodynamic_state(rho, temperature)

        self.assertGreaterEqual(float(state.density.min().item()), config.min_density * (1.0 - 1e-6))
        self.assertGreaterEqual(float(state.temperature.min().item()), config.min_temperature * (1.0 - 1e-6))
        self.assertGreaterEqual(float(state.pressure.min().item()), config.min_pressure * (1.0 - 1e-6))

    def test_factory_preserves_raw_solver_when_thermal_disabled(self):
        cfg = make_config(0.2, thermal_enabled=False)

        solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)

        self.assertIsInstance(solver, D3Q27CascadedSolver)
        self.assertNotIsInstance(solver, ThermalD3Q27Solver)

    def test_thermal_wrapper_adds_experimental_mach2_metadata(self):
        cfg = make_config(2.0, thermal_enabled=True)
        device = torch.device("cpu")
        solver = create_thermal_lbm_solver(cfg, device, TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32, device=device)
        geometry[3:5, 3:5, 3:5] = 1.0

        solver.collide_stream(geometry, steps=1)
        results = solver.compute_aerodynamic_coefficients(geometry)

        self.assertIsInstance(solver, ThermalD3Q27Solver)
        self.assertEqual(results["thermal_model"], "coupled_d3q7_temperature_bgk")
        self.assertEqual(results["compressibility_model"], "staged_thermal_lbm_experimental")
        self.assertEqual(results["pressure_model"], "ideal_gas_diagnostic_p_equals_rho_R_T")
        self.assertEqual(results["validity_regime"], "experimental_thermal_lbm_unvalidated")
        self.assertEqual(results["claim_grade"], "no_claim_experimental")
        self.assertEqual(results["training_drag_source"], "none_thermal_internal_lbm_unvalidated")
        self.assertFalse(results["pinn_ready"])
        self.assertFalse(results["shock_capable"])
        self.assertEqual(results["thermodynamic_solver"], "staged_d3q7_thermal_bgk_attached_to_d3q27")
        self.assertTrue(math.isfinite(results["temperature_mean"]))
        self.assertTrue(math.isfinite(results["thermodynamic_pressure_mean"]))
        self.assertIn("flow_isothermal_pressure_sum", results)
        self.assertIn("thermodynamic_pressure_sum", results)
        self.assertAlmostEqual(results["pressure_sum"], results["flow_isothermal_pressure_sum"], places=5)
        self.assertTrue(torch.allclose(solver.pressure, solver.flow_solver.pressure))
        self.assertFalse(torch.allclose(solver.pressure, solver.thermodynamic_pressure))

    def test_pressure_gradient_coupling_produces_clipped_guo_force(self):
        cfg = make_config(0.5, thermal_enabled=True)
        cfg.thermal_lbm_config = ThermalLBMConfig(
            reference_temperature=300.0,
            pressure_coupling_strength=0.1,
            pressure_gradient_clip=0.02,
        )
        solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
        ramp = torch.linspace(300.0, 600.0, solver.resolution).view(-1, 1, 1).expand(
            solver.resolution,
            solver.resolution,
            solver.resolution,
        )
        solver.thermal_solver.set_temperature(ramp)

        force = solver.compute_thermal_pressure_force()

        self.assertEqual(force.shape, (3, solver.resolution, solver.resolution, solver.resolution))
        self.assertGreater(float(torch.abs(force[0]).max().item()), 0.0)
        self.assertLess(float(torch.abs(force[1]).max().item()), 1e-8)
        self.assertLess(float(torch.abs(force[2]).max().item()), 1e-8)
        self.assertLessEqual(float(torch.abs(force).max().item()), 0.020001)

    def test_pressure_gradient_coupling_adds_to_user_ext_force_without_mutation(self):
        cfg = make_config(0.5, thermal_enabled=True)
        cfg.thermal_lbm_config = ThermalLBMConfig(
            reference_temperature=300.0,
            pressure_coupling_strength=0.1,
            pressure_gradient_clip=0.02,
        )
        solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        ramp = torch.linspace(300.0, 600.0, solver.resolution).view(-1, 1, 1).expand(
            solver.resolution,
            solver.resolution,
            solver.resolution,
        )
        solver.thermal_solver.set_temperature(ramp)
        caller_force = torch.full((3, 8, 8, 8), 0.001, dtype=torch.float32)
        caller_force_before = caller_force.clone()
        captured = {}

        def fake_flow_step(_geometry_mask, steps=100, ext_force=None):
            captured["ext_force"] = ext_force.clone()

        solver.flow_solver.collide_stream = fake_flow_step
        solver.collide_stream(geometry, steps=1, ext_force=caller_force)

        self.assertTrue(torch.allclose(caller_force, caller_force_before))
        self.assertTrue(torch.allclose(captured["ext_force"], caller_force_before + solver.thermal_pressure_gradient_force))

    def test_disabled_pressure_coupling_passes_user_ext_force_through(self):
        cfg = make_config(0.5, thermal_enabled=True)
        cfg.thermal_lbm_config = ThermalLBMConfig(
            reference_temperature=300.0,
            pressure_coupling_strength=0.0,
        )
        solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        caller_force = torch.full((3, 8, 8, 8), 0.001, dtype=torch.float32)
        captured = {}

        def fake_flow_step(_geometry_mask, steps=100, ext_force=None):
            captured["ext_force"] = ext_force

        solver.flow_solver.collide_stream = fake_flow_step
        solver.collide_stream(geometry, steps=1, ext_force=caller_force)

        self.assertIs(captured["ext_force"], caller_force)

    def test_staged_features_do_not_promote_high_mach_claims(self):
        cfg = make_config(2.0, thermal_enabled=True)
        cfg.thermal_lbm_config = ThermalLBMConfig(
            reference_temperature=310.0,
            inlet_temperature=350.0,
            pressure_coupling_strength=0.1,
            pressure_gradient_clip=0.02,
            shock_stabilization_enabled=True,
        )
        solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[3:5, 3:5, 3:5] = 1.0

        solver.collide_stream(geometry, steps=1)
        results = solver.compute_aerodynamic_coefficients(geometry)

        self.assertEqual(results["claim_grade"], "no_claim_experimental")
        self.assertEqual(results["validity_regime"], "experimental_thermal_lbm_unvalidated")
        self.assertFalse(results["pinn_ready"])
        self.assertFalse(results["shock_capable"])
        self.assertIn("shock_sensor_max", results)
        self.assertIn("shock_cell_count", results)
        self.assertEqual(results["thermal_boundary_model"], "fixed_temperature_inlet_zero_gradient_outlet")
        self.assertEqual(results["compressible_boundary_status"], "staged_thermal_boundary_not_characteristic_validated")
        self.assertEqual(results["inlet_outlet_regime"], "supersonic_experimental")
        self.assertEqual(results["thermal_force_coupling"], "pressure_gradient_guo_forcing_experimental")
        self.assertIn("thermal_pressure_gradient_force_norm", results)

    def test_disabled_shock_stabilization_metadata_reports_disabled(self):
        cfg = make_config(2.0, thermal_enabled=True)
        cfg.thermal_lbm_config = ThermalLBMConfig(
            reference_temperature=310.0,
            shock_stabilization_enabled=False,
            pressure_coupling_strength=0.0,
        )
        solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[3:5, 3:5, 3:5] = 1.0

        solver.collide_stream(geometry, steps=1)
        results = solver.compute_aerodynamic_coefficients(geometry)

        self.assertEqual(results["shock_stabilization_model"], "disabled")
        self.assertEqual(results["thermal_force_coupling"], "diagnostic_pressure_not_force_coupled")
        self.assertFalse(results["shock_capable"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA thermal wrapper path requires a CUDA device")
    def test_cuda_wrapper_accepts_cpu_geometry_and_steps_thermal_path(self):
        cfg = make_config(2.0, thermal_enabled=True)
        solver = create_thermal_lbm_solver(cfg, torch.device("cuda"), TestPhysicsConfig)
        geometry_cpu = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry_cpu[3:5, 3:5, 3:5] = 1.0

        solver.collide_stream(geometry_cpu, steps=1)
        results = solver.compute_aerodynamic_coefficients(geometry_cpu)

        self.assertEqual(solver.thermal_solver.g.device.type, "cuda")
        self.assertEqual(results["thermal_solver_device"], "cuda")
        self.assertTrue(math.isfinite(results["thermodynamic_pressure_mean"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA thermal coupling path requires a CUDA device")
    def test_cuda_cpu_ext_force_is_coerced_before_thermal_addition(self):
        cfg = make_config(0.5, thermal_enabled=True)
        cfg.thermal_lbm_config = ThermalLBMConfig(
            reference_temperature=300.0,
            pressure_coupling_strength=0.1,
            pressure_gradient_clip=0.02,
        )
        solver = create_thermal_lbm_solver(cfg, torch.device("cuda"), TestPhysicsConfig)
        geometry_cpu = torch.zeros((8, 8, 8), dtype=torch.float32)
        ramp = torch.linspace(300.0, 600.0, solver.resolution).view(-1, 1, 1).expand(
            solver.resolution,
            solver.resolution,
            solver.resolution,
        )
        solver.thermal_solver.set_temperature(ramp)
        caller_force = torch.full((3, 8, 8, 8), 0.001, dtype=torch.float32)
        caller_force_before = caller_force.clone()
        captured = {}

        def fake_flow_step(_geometry_mask, steps=100, ext_force=None):
            captured["ext_force"] = ext_force.clone()

        solver.flow_solver.collide_stream = fake_flow_step
        solver.collide_stream(geometry_cpu, steps=1, ext_force=caller_force)

        self.assertEqual(captured["ext_force"].device.type, "cuda")
        self.assertEqual(solver.thermal_pressure_gradient_force.device.type, "cuda")
        self.assertTrue(torch.allclose(caller_force, caller_force_before))
        self.assertTrue(torch.allclose(captured["ext_force"].cpu(), caller_force_before + solver.thermal_pressure_gradient_force.cpu()))

    def test_mach_mapping_helper_is_unchanged_by_thermal_path(self):
        self.assertAlmostEqual(mach_to_lattice_velocity(2.0), 2.0 / math.sqrt(3.0), places=12)


if __name__ == "__main__":
    unittest.main()
