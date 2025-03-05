import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import cProfile

def create_grid(nx, ny, aspect_ratio=1.0):
    """Create 2D grid"""
    x = np.linspace(0, aspect_ratio, nx)
    y = np.linspace(0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = aspect_ratio / (nx - 1)
    dy = 1.0 / (ny - 1)
    return X, Y, dx, dy

@jit(nopython=True)
def initialize_fields(nx, ny):
    """Initialize temperature, streamfunction, and vorticity fields"""
    T = np.zeros((nx, ny))
    psi = np.zeros((nx, ny))
    omega = np.zeros((nx, ny))
    
    # Create proper 2D array for temperature initialization
    y = np.linspace(0, 1, ny)
    T = 1.0 - y[np.newaxis, :]  # Makes a 2D array properly
    T = np.broadcast_to(T, (nx, ny)).copy()  # Make it writable
    T += 0.01 * np.random.randn(nx, ny)  # Add perturbation
    
    return T, psi, omega

@jit(nopython=True)
def apply_boundary_conditions(T, psi):
    """Apply boundary conditions for temperature and streamfunction"""
    # Temperature BCs
    T[:, 0] = 1.0  # Bottom (hot)
    T[:, -1] = 0.0  # Top (cold)
    T[0, :] = T[1, :]  # Adiabatic sides
    T[-1, :] = T[-2, :]
    
    # Streamfunction BCs (no-slip)
    psi[0, :] = psi[-1, :] = 0
    psi[:, 0] = psi[:, -1] = 0
    
    return T, psi

@jit(nopython=True)
def solve_poisson(omega, psi, dx, dy, omega_sor=1.3, tolerance=1e-6, max_iter=1000):
    """Solve ∇²ψ = -ω using SOR with explicit Gauss-Seidel ordering."""
    nx, ny = omega.shape
    dx2, dy2 = dx*dx, dy*dy

    solution = psi.copy()

    for iter in range(max_iter):
        old_solution = solution.copy()  # For convergence check

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # The core SOR update
                solution[i, j] = (1 - omega_sor) * solution[i, j] + \
                                omega_sor / (2/dx2 + 2/dy2) * (
                                    (old_solution[i+1, j] + solution[i-1, j]) / dx2 +
                                    (old_solution[i, j+1] + solution[i, j-1]) / dy2 +
                                    omega[i, j]
                                )

        # Enforce boundary conditions
        solution[0, :] = solution[-1, :] = 0
        solution[:, 0] = solution[:, -1] = 0

        # Convergence check
        max_change = np.max(np.abs(solution - old_solution))
        if max_change < tolerance:
      #      print(f"Converged in {iter+1} iterations.")
            break

    return solution

@jit(nopython=True)
def calculate_velocities(psi, dx, dy):
    """Calculate velocity components from streamfunction"""
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dx)
    
    return u, v

@jit(nopython=True)
def advance_temperature(T, u, v, dx, dy, dt):
    """Advance temperature field one timestep"""
    T_new = T.copy()
    
    # Advection-diffusion
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] - dt * (
        u[1:-1, 1:-1] * (T[2:, 1:-1] - T[:-2, 1:-1]) / (2*dx) +
        v[1:-1, 1:-1] * (T[1:-1, 2:] - T[1:-1, :-2]) / (2*dy)
    ) + dt * (
        (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dx**2) +
        (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / (dy**2)
    )
    
    return T_new

@jit(nopython=True)
def update_vorticity(T, Ra, dx):
    """Update vorticity based on temperature gradient"""
    omega = np.zeros_like(T)
 #   omega[1:-1, 1:-1] = Ra * (T[2:, 1:-1] - T[:-2, 1:-1]) / (2*dx)
    omega[1:-1, 1:-1] = Ra * (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx) 
    return omega

def plot_fields(X, Y, T, psi):
    """Plot temperature and streamfunction fields"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.pcolormesh(X, Y, T, cmap='RdBu_r')
    ax1.set_aspect('equal')
    ax1.set_title('Temperature')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.pcolormesh(X, Y, psi, cmap='RdBu')
    ax2.set_aspect('equal')
    ax2.set_title('Streamfunction')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

@jit(nopython=True)
def calculate_dt(u, v, dx, dy, dt_max, cfl_target=0.5, INF_VAL = 1e10):
    """Calculate timestep based on CFL condition."""
    
    #Maximum velocities
    u_max = np.max(np.abs(u))
    v_max = np.max(np.abs(v))

    #CFL condition
    dt_x = dx / u_max if u_max > 0 else INF_VAL
    dt_y = dy / v_max if v_max > 0 else INF_VAL

    dt = cfl_target * min(dt_x, dt_y)
    return min(dt, dt_max)  #Limit by dt_max

@jit(nopython=True)
def simulate_step(T, psi, omega, Ra, dx, dy, dt, INF_VAL = 1e10):
  
    """Advance simulation by one timestep"""
    # Update vorticity from temperature
    omega = update_vorticity(T, Ra, dx)
    
    # Solve for streamfunction
    psi = solve_poisson(omega, psi, dx, dy)
    
    # Calculate velocities
    u, v = calculate_velocities(psi, dx, dy)
    
    # Update temperature
    T = advance_temperature(T, u, v, dx, dy, dt)
    
    # Apply boundary conditions
    T, psi = apply_boundary_conditions(T, psi)
    
    
    
    return T, psi, omega, u, v

def main():
    # Simulation parameters
    nx, ny = 64, 64
    Ra = 2e3
    dt = 0.001  #Initial timestep
    dt_max = 0.01 #Maximum timestep
    cfl_target = 0.1 #Target CFL number
    n_steps = 2000
    plot_interval = 100
    INF_VAL = 1e10
    
    # Setup
    X, Y, dx, dy = create_grid(nx, ny)
    T, psi, omega = initialize_fields(nx, ny)
    T, psi = apply_boundary_conditions(T, psi)
    
    # Main loop
   # start_time = time.time()  # Start time
    for step in range(n_steps):
        T, psi, omega, u, v = simulate_step(T, psi, omega, Ra, dx, dy, dt, INF_VAL)

        #Adaptive time stepping
        dt = calculate_dt(u, v, dx, dy, dt_max, cfl_target, INF_VAL)
        
        if step % plot_interval == 0:
            print(f"Step {step}, dt = {dt}")
            plot_fields(X, Y, T, psi)
    #end_time = time.time()  # End time
    #elapsed_time = end_time - start_time
  #  print(f" simulate_total time: {elapsed_time:.2f} seconds")    
            
if __name__ == "__main__":
    main()
