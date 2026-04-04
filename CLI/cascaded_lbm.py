import torch


class CascadedLBM:
    """Central moments cascaded collision"""

    @staticmethod
    def compute_central_moments(f, ux, uy, uz, ex, ey, ez):
        """Transform populations to central moments"""
        K = {}

        # Order 0: density
        K['000'] = torch.sum(f, dim=0)

        # For higher moments, loop over directions
        for moment_key, moment_powers in [
            ('100', (1, 0, 0)), ('010', (0, 1, 0)), ('001', (0, 0, 1)),
            ('200', (2, 0, 0)), ('020', (0, 2, 0)), ('002', (0, 0, 2)),
            ('110', (1, 1, 0)), ('101', (1, 0, 1)), ('011', (0, 1, 1)),
            ('300', (3, 0, 0)), ('030', (0, 3, 0)), ('003', (0, 0, 3)),
            ('210', (2, 1, 0)), ('201', (2, 0, 1)), ('120', (1, 2, 0)),
            ('021', (0, 2, 1)), ('102', (1, 0, 2)), ('012', (0, 1, 2)),
            ('111', (1, 1, 1)), ('220', (2, 2, 0)), ('202', (2, 0, 2)),
            ('022', (0, 2, 2)), ('211', (2, 1, 1)), ('121', (1, 2, 1)),
            ('400', (4, 0, 0)), ('040', (0, 4, 0)), ('004', (0, 0, 4)),
            ('310', (3, 1, 0)), ('301', (3, 0, 1)), ('130', (1, 3, 0)),
            ('031', (0, 3, 1)), ('103', (1, 0, 3)), ('013', (0, 1, 3)),
            ('220', (2, 2, 0)), ('202', (2, 0, 2)), ('022', (0, 2, 2)),
            ('311', (3, 1, 1)), ('131', (1, 3, 1)), ('113', (1, 1, 3)) ]:
            px, py, pz = moment_powers
            moment = torch.zeros_like(ux)

            for i in range(len(ex)):
                cx = ex[i] - ux
                cy = ey[i] - uy
                cz = ez[i] - uz
                moment += f[i] * (cx**px) * (cy**py) * (cz**pz)

            K[moment_key] = moment

        return K

    @staticmethod
    def equilibrium_central_moments(rho, cs2=1/3):
        """Equilibrium central moments"""
        K_eq = {}
        K_eq['000'] = rho
        K_eq['100'] = K_eq['010'] = K_eq['001'] = torch.zeros_like(rho)
        K_eq['200'] = K_eq['020'] = K_eq['002'] = rho * cs2
        K_eq['110'] = K_eq['101'] = K_eq['011'] = torch.zeros_like(rho)
        K_eq['111'] = torch.zeros_like(rho)
        K_eq['300'] = K_eq['030'] = K_eq['003'] = torch.zeros_like(rho)
        K_eq['210'] = K_eq['201'] = K_eq['120'] = torch.zeros_like(rho)
        K_eq['021'] = K_eq['102'] = K_eq['012'] = torch.zeros_like(rho)
        K_eq['400'] = K_eq['040'] = K_eq['004'] = torch.zeros_like(rho)
        K_eq['310'] = K_eq['301'] = K_eq['130'] = torch.zeros_like(rho)
        K_eq['031'] = K_eq['103'] = K_eq['013'] = torch.zeros_like(rho)
        K_eq['311'] = K_eq['131'] = K_eq['113'] = torch.zeros_like(rho)
        return K_eq

    @staticmethod
    def cascaded_relax(K, K_eq, s_nu, s_e, s_h):
        K_post = {}

        # Step 1: Conserve mass and momentum
        K_post['000'] = K['000']
        K_post['100'] = K['100']
        K_post['010'] = K['010']
        K_post['001'] = K['001']

        # Step 2: Relax energy
        K_post['200'] = K['200'] + s_e * (K_eq['200'] - K['200'])
        K_post['020'] = K['020'] + s_e * (K_eq['020'] - K['020'])
        K_post['002'] = K['002'] + s_e * (K_eq['002'] - K['002'])

        # Step 3: Relax stress (viscosity)
        K_post['110'] = K['110'] + s_nu * (K_eq['110'] - K['110'])
        K_post['101'] = K['101'] + s_nu * (K_eq['101'] - K['101'])
        K_post['011'] = K['011'] + s_nu * (K_eq['011'] - K['011'])

        # Step 4: Relax ALL higher order moments
        higher_moments = ['111', '300', '030', '003', '210', '201', '120', '021', '102', '012',
                        '220', '202', '022', '211', '121',
                        '400', '040', '004', '310', '301', '130', '031', '103', '013',
                        '311', '131', '113']

        for moment in higher_moments:
            if moment in K:  # Only relax if it exists
                K_eq_val = K_eq.get(moment, torch.zeros_like(K[moment]))
                K_post[moment] = K[moment] + s_h * (K_eq_val - K[moment])

        return K_post

    @staticmethod
    def moments_to_populations(K, ux, uy, uz, ex, ey, ez, w):
        """Inverse transform: central moments → populations"""
        rho = K['000']
        n_dirs = len(ex)
        f = torch.zeros(n_dirs, *rho.shape, device=rho.device)

        # Compute equilibrium base
        u_sq = ux**2 + uy**2 + uz**2

        for i in range(n_dirs):
            cu = ex[i]*ux + ey[i]*uy + ez[i]*uz

            # Equilibrium part
            feq = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

            # Non-equilibrium corrections from central moments
            # Shift to moving frame
            cx = ex[i] - ux
            cy = ey[i] - uy
            cz = ez[i] - uz

            # Add non-equilibrium stress corrections (2nd order)
            fneq = w[i] * (
                # Diagonal stress
                0.5 * (K['200'] - rho/3) * (cx**2 - 1/3) +
                0.5 * (K['020'] - rho/3) * (cy**2 - 1/3) +
                0.5 * (K['002'] - rho/3) * (cz**2 - 1/3) +
                # Off-diagonal stress
                K['110'] * cx * cy +
                K['101'] * cx * cz +
                K['011'] * cy * cz
            )

            f[i] = feq + fneq

        return f
