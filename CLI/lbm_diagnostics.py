import torch


def _spacing_3d(spacing):
    if spacing is None:
        return None
    return (spacing, spacing, spacing)


def compute_velocity_gradients(ux, uy, uz, spacing=None):
    """Compute the full velocity-gradient tensor once."""
    grad_spacing = _spacing_3d(spacing)
    return (
        torch.gradient(ux, dim=(0, 1, 2), spacing=grad_spacing),
        torch.gradient(uy, dim=(0, 1, 2), spacing=grad_spacing),
        torch.gradient(uz, dim=(0, 1, 2), spacing=grad_spacing),
    )


def compute_strain_rate_tensor(ux, uy, uz, spacing=None, gradients=None):
    """Compute strain rate tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i)."""
    if gradients is None:
        gradients = compute_velocity_gradients(ux, uy, uz, spacing=spacing)
    grad_ux, grad_uy, grad_uz = gradients
    dux_dx, dux_dy, dux_dz = grad_ux
    duy_dx, duy_dy, duy_dz = grad_uy
    duz_dx, duz_dy, duz_dz = grad_uz

    S11 = dux_dx.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
    S22 = duy_dy.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
    S33 = duz_dz.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
    S12 = 0.5 * (dux_dy + duy_dx)
    S13 = 0.5 * (dux_dz + duz_dx)
    S23 = 0.5 * (duy_dz + duz_dy)

    return S11, S22, S33, S12, S13, S23


def compute_vorticity(ux, uy, uz, spacing=None, gradients=None):
    """Compute vorticity omega = curl(u)."""
    if gradients is None:
        gradients = compute_velocity_gradients(ux, uy, uz, spacing=spacing)
    grad_ux, grad_uy, grad_uz = gradients
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
