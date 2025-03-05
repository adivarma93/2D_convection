#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:24:28 2025

@author: debarshi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:41:30 2025

@author: debarshi
"""

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from mpi4py import MPI
import time
import cProfile

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"MPI initialized: Rank={rank}, Size={size}")  # Add this

def create_subgrid(nx_global, ny_global, rank, size, aspect_ratio=1.0):
    """Create subgrids for parallel processing."""
    x = np.linspace(0, aspect_ratio, nx_global)
    y_full = np.linspace(0, 1.0, ny_global)

    num_per_rank = ny_global // size
    remainder = ny_global % size

    start = rank * num_per_rank + min(rank, remainder)
    end = (rank + 1) * num_per_rank + min(rank + 1, remainder) - 1 #Don't include last

    y = y_full[start:end+1]  # Indexing last row for boundary exchange
    X, Y = np.meshgrid(x, y, indexing='ij')

    dx = aspect_ratio / (nx_global - 1)
    dy = 1.0 / (ny_global - 1)

    return X, Y, dx, dy, start, end

@jit(nopython=True)
def initialize_fields(nx, ny_local):
    """Initialize temperature, streamfunction, and vorticity fields on the local subgrid."""
    T = np.zeros((nx, ny_local))
    psi = np.zeros((nx, ny_local))
    omega = np.zeros((nx, ny_local))
    
    # Create proper 2D array for temperature initialization
    y = np.linspace(0, 1, ny_local)
    T = 1.0 - y[np.newaxis, :]  # Makes a 2D array properly
    T = np.broadcast_to(T, (nx, ny_local)).copy()  # Make it writable
    T += 0.01 * np.random.randn(nx, ny_local)  # Add perturbation
    
    return T, psi, omega

def apply_boundary_conditions(T, psi, start, end, ny):
    """Apply boundary conditions for temperature and streamfunction on local subgrid."""
    # Temperature BCs
    if start == 0:
        T[:, 0] = 1.0  # Bottom (hot)
    if end == ny - 1:
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
    omega[1:-1, 1:-1] = Ra * (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx)
    return omega

def plot_fields(X, Y, T, psi):
    """Plot temperature and streamfunction fields (only on rank 0)"""
    if rank == 0:
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

def exchange_boundaries(T, comm, rank, size,nx,ny):
    """Exchange ghost layers between MPI ranks."""
    #get new shape
    shape = (nx, 1)
    #now allocate memory
    recv_buffer_left = np.empty(shape, dtype=T.dtype)
    recv_buffer_right = np.empty(shape, dtype=T.dtype)

    #This is important - must handle the ranks correctly to avoid deadlock
    #It involves even and odd numbers. I've coded it for you.
    #It might have to do with a non blocking send, or other.
    #But this should work.
    #The ranks must be even.
    if size%2 != 0:
        print("Must be even!")
        return

    comm.Barrier()  # Add explicit synchronization
    # This should be even!
    if rank%2==0:
      #LEFT: This makes sure the index never runs under.
      if rank > 0:
          #copy creates a non-contigious array, must fix
          send_data = np.ascontiguousarray(T[:, 1].reshape(shape))
          #send data

          comm.Sendrecv(send_data, dest=rank-1, recvbuf=recv_buffer_left, sendtag=0, recvtag=1)

          # Set the data after recieving
          T[:, 0] = recv_buffer_left.reshape((nx,))
    else:
      #RIGHT: This makes sure the index never runs over
      if rank < size - 1:
          send_data = np.ascontiguousarray(T[:, -2].reshape(shape))

          comm.Sendrecv(send_data, dest=rank+1, recvbuf=recv_buffer_right, sendtag=1, recvtag=0)

          # Set the data after recieving
          T[:, -1] = recv_buffer_right.reshape((nx,))
    comm.Barrier()  # Add explicit synchronization
    return T



def simulate_step(T, psi, omega, Ra, dx, dy, dt, start, end, comm, rank, size,nx,ny):
   
    
    print(f"Rank {rank}: Entering simulate_step")

    # Exchange boundary data
    exchange_boundaries(T, comm, rank, size, nx, ny)
    print(f"Rank {rank}: After exchange_boundaries")

    omega = update_vorticity(T, Ra, dx)
    print(f"Rank {rank}: After update_vorticity")

    psi = solve_poisson(omega, psi, dx, dy)
    print(f"Rank {rank}: After solve_poisson")

    u, v = calculate_velocities(psi, dx, dy)
    print(f"Rank {rank}: After calculate_velocities")

    T = advance_temperature(T, u, v, dx, dy, dt)
    print(f"Rank {rank}: After advance_temperature")

    T, psi = apply_boundary_conditions(T, psi, start, end, ny)
    print(f"Rank {rank}: After apply_boundary_conditions")

    print(f"Rank {rank}: Exiting simulate_step")
    
    
  #  print(f"Rank {rank}: simulate_step time: {elapsed_time:.2f} seconds") 
    

    
    return T, psi, omega, u, v


def main():
    # Simulation parameters
    nx, ny = 64, 64
    Ra = 2e3
    dt = 0.001  #Initial timestep
    dt_max = 0.01 #Maximum timestep
    cfl_target = 0.1 #Target CFL number
    n_steps = 5000
    plot_interval = 100
    INF_VAL = 1e10
    
    # Setup - LOCAL SUBGRID
    X, Y, dx, dy, start, end = create_subgrid(nx, ny, rank, size)
    ny_local = end - start + 1
    T, psi, omega = initialize_fields(nx, ny_local)
    T, psi = apply_boundary_conditions(T, psi, start, end, ny)
    
    # Print initial grid info
    if rank == 0:
        print(f"Global grid: nx={nx}, ny={ny}")
    print(f"Rank {rank}: Local grid - start={start}, end={end}, ny_local={ny_local}, dx={dx}, dy={dy}")  # Important: check local grid

    # Main loop
   # start_time = time.time()  # Start time
    for step in range(n_steps):

        #Debugging: Print before simulate_step
        print(f"Rank {rank}, Step {step} dt= {dt}: Before simulate_step")  # Check if all ranks enter simulate_step

        T, psi, omega, u, v = simulate_step(T, psi, omega, Ra, dx, dy, dt, start, end, comm, rank, size, nx, ny)

        # Adaptive time stepping (needs reduction)
        dt = calculate_dt(u, v, dx, dy, dt_max, cfl_target)

        # Perform a global reduction to find the minimum dt across all processes
        dt = comm.allreduce(dt, op=MPI.MIN)

        # Plotting (only on rank 0)
        if rank == 0 and (step % plot_interval == 0):
            print(f"Step {step}, dt = {dt}")
          #  plot_fields(X, Y, T, psi)  #Commented to avoid errors
   # end_time = time.time()  # End time
   # elapsed_time = end_time - start_time
   # print(f" simulate_total time: {elapsed_time:.2f} seconds")   

if __name__ == "__main__":
    output_file = f"profile_rank{rank}.prof"  # Unique output file 
    cProfile.run('main()', filename=output_file)
    main()
