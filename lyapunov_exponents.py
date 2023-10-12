import numpy as np

def jacobian(func, r_state, delta=1e-6):
    """
    Compute the Jacobian matrix of a given function at a specific state.
    """
    dim = len(r_state)
    J = np.zeros((dim, dim))
    for i in range(dim):
        perturbation = np.zeros(dim)
        perturbation[i] = delta
        J[:, i] = (func(r_state + perturbation) - func(r_state - perturbation)) / (2 * delta)
    return J

def compute_lyapunov_exponents(model, u_init, iterations=1000, delta=1e-6):
    dim = model.dim_reservoir
    w = np.eye(dim)
    lyapunov_exponents = np.zeros(dim)
    
    r_state = model.advance_r_state(u_init)
    
    for _ in range(iterations):
        # Compute Jacobian
        J = jacobian(lambda r_state: model.advance_r_state(r_state), r_state, delta)
        
        # Evolve the perturbations using the Jacobian
        w = np.dot(J, w)
        
        # Orthonormalize using Gram-Schmidt procedure
        for i in range(dim):
            w[:, i] = w[:, i] - sum(np.dot(w[:, j], w[:, i]) for j in range(i)) * w[:, j]
            norm = np.linalg.norm(w[:, i])
            w[:, i] = w[:, i] / norm
            lyapunov_exponents[i] += np.log(norm)
        
        # Advance the main system
        r_state = model.advance_r_state(u_init)
    
    lyapunov_exponents = lyapunov_exponents / iterations
    return lyapunov_exponents