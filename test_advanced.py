import torch
import sys
import os
sys.path.append('CLI')
from advanced_lbm_solver import GPULBMSolver
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig

device = torch.device('cpu')
config = CFDConfig(base_grid_resolution=32, mach_number=0.01, reynolds_number=1000)
config.lbm_config = LBMPhysicsConfig()
config.lbm_config.grid_spacing = 1.0 / 32.0

solver = GPULBMSolver(config, device, LBMPhysicsConfig)
geometry_mask = torch.zeros((32, 32, 32), device=device)
geometry_mask[14:18, 14:18, 14:18] = 1.0

solver.collide_stream(geometry_mask, steps=100)
print(f"force_samples: {solver.force_samples}")
print(f"force_x_accum: {solver.force_x_accum}")
print(f"u_lat: {solver.u_lat}")
coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)
print(f"Drag Coeff: {coeffs['drag_coefficient']}")
