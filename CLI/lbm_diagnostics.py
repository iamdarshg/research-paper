import torch


def compute_strain_rate_tensor(ux, uy, uz):
    """Compute strain rate tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i)."""
    dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2))
    duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2))
    duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2))

    S11 = dux_dx.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
    S22 = duy_dy.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
    S33 = duz_dz.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
    S12 = 0.5 * (dux_dy + duy_dx)
    S13 = 0.5 * (dux_dz + duz_dx)
    S23 = 0.5 * (duy_dz + duz_dy)

    return S11, S22, S33, S12, S13, S23


def compute_vorticity(ux, uy, uz):
    """Compute vorticity omega = curl(u)."""
    grad_ux = torch.gradient(ux, dim=(0, 1, 2))
    grad_uy = torch.gradient(uy, dim=(0, 1, 2))
    grad_uz = torch.gradient(uz, dim=(0, 1, 2))

    dux_dx, dux_dy, dux_dz = grad_ux
    duy_dx, duy_dy, duy_dz = grad_uy
    duz_dx, duz_dy, duz_dz = grad_uz

    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy

    return (
        omega_x.nan_to_num(1e-12, posinf=1e18, neginf=-1e18),
        omega_y.nan_to_num(1e-12, posinf=1e18, neginf=-1e18),
        omega_z.nan_to_num(1e-12, posinf=1e18, neginf=-1e18),
    )
