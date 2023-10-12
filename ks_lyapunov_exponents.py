import numpy as np
from scipy.linalg import qr
from data import ks_pseudospectral

def compute_lyapunov_exponents(u0, L, N, dt, T, k, perturb=1e-12):
    d = N  # Dimension of the system
    Q = np.eye(d)  # Orthogonal matrix
    R_diag_list = []  # To store the diagonal of R

    for _ in range(k):
        # Compute the trajectory of the main system using ks_pseudospectral
        # This replaces the MATLAB integrator
        traj = [u0]
        u = u0.copy()
        for t in np.arange(0, T, dt):
            u += dt * ks_pseudospectral(u, t, L, N)
            traj.append(u.copy())
        traj = np.array(traj)

        phi_cols = []
        for j in range(d):
            # Perturb the initial condition
            delta_u = perturb * Q[:, j]
            u_pert = u0 + delta_u

            # Compute the trajectory of the perturbed system
            pert_traj = [u_pert]
            u = u_pert.copy()
            for t in np.arange(0, T, dt):
                u += dt * ks_pseudospectral(u, t, L, N)
                pert_traj.append(u.copy())
            pert_traj = np.array(pert_traj)

            # Compute the difference between perturbed and main trajectory
            phi_cols.append((pert_traj[-1] - traj[-1]) / perturb)
        phi = np.stack(phi_cols, axis=1)

        # QR decomposition
        Q, R = qr(phi)
        R_diag_list.append(np.diag(R))

    # Compute Lyapunov exponents
    R_diag_arr = np.stack(R_diag_list)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T

    return lyapunov_exps
