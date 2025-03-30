
# 2D Rayleigh-Bénard Convection Simulator

This repository contains two Python scripts to simulate **Rayleigh-Bénard convection**, a fundamental fluid dynamics phenomenon where a fluid layer, heated from below and cooled from above, develops buoyancy-driven flow patterns. The simulations model the 2D Boussinesq approximation of the Navier-Stokes equations.

- **`2Dconvection_numba.py`**: A serial implementation optimized for single-threaded execution.
- **`2Dconvection_numba_mpi.py`**: A parallel implementation using MPI for distributed computing.

## Overview

Both scripts simulate Rayleigh-Bénard convection by solving equations for temperature, vorticity, and streamfunction. The serial version runs on a single process, while the MPI version distributes the computational domain across multiple processes for scalability.

### Physics Modeled
- **Temperature evolution**: Advection and diffusion of heat.
- **Vorticity**: Driven by horizontal temperature gradients (buoyancy).
- **Streamfunction**: Represents incompressible flow, solved via ∇²ψ = -ω.
- **Boundary conditions**: Hot bottom (T = 1), cold top (T = 0), adiabatic sides; no-slip for streamfunction (ψ = 0).


## Files

### 1. `2Dconvection_numba.py` (Serial Version)
- **Description**: A single-process implementation optimized with **Numba** for performance.
- **Numerical Methods**:
  - Finite difference method with explicit time-stepping.
  - Successive Over-Relaxation (SOR) for the Poisson equation.
  - Adaptive timestepping based on the CFL condition.
- **Features**:
  - Visualizes temperature and streamfunction fields using Matplotlib.
  - Suitable for small-to-medium grids or educational use.

### 2. `2Dconvection_numba_mpi.py` (Parallel Version)
- **Description**: A parallel implementation using **MPI** (via `mpi4py`) to split the domain along the y-axis across multiple processes.
- **Numerical Methods**:
  - Same as the serial version, with domain decomposition.
  - Boundary data exchange between adjacent processes using MPI Sendrecv.
  - Global timestep synchronization via MPI reduction.
- **Features**:
  - Scales to larger grids using multiple CPU cores.
  - Includes profiling with `cProfile` per rank.
  - Plotting is disabled by default (uncomment for rank 0 testing).

## Requirements

### Serial Version
- Python 3.x
- NumPy
- Matplotlib
- Numba

Install with:
```bash
pip install numpy matplotlib numba
```

### Parallel Version 
- mpi4py
- An MPI implementation (e.g., MPICH or OpenMPI)

Install with:
```bash
pip install mpi4py
```
On Ubuntu:
```bash
sudo apt-get install mpich
```

## Usage

### Running the Serial Version
1. Clone the repository:
   ```bash
   git clone https://github.com/adivarma93/2D_convection.git
   cd 2D_convection
   ```
2. Run the script:
   ```bash
   python 2Dconvection_numba.py
   ```
3. Adjust parameters in `main()`:
   - `nx`, `ny`: Grid resolution (e.g., 64x64).
   - `Ra`: Rayleigh number (e.g., 2e3).
   - `n_steps`: Number of timesteps.
   - `plot_interval`: Frequency of visualization.

### Running the Parallel Version
1. Ensure MPI and `mpi4py` are installed.
2. Run the script with `mpiexec` or `mpirun`:
   ```bash
   mpiexec -n 4 python 2Dconvection_numba_mpi.py
   ```
   - Replace `4` with the desired number of processes (must be even due to boundary exchange logic).
3. Adjust parameters in `main()`:
   - Same as the serial version; domain is split along `ny`.
4. Check output:
   - Each rank prints progress and generates a profiling file (`profile_rankX.prof`).
   - Uncomment `plot_fields` in `main()` for visualization on rank 0 (for testing).

#### Analyzing Profiles
View profiling data for a rank (e.g., rank 0):
```bash
python -m pstats profile_rank0.prof
```
Use commands like `sort time` and `stats` in the interactive prompt.

## Example Output

- **Serial**: For Ra = 2000, nx = ny = 64, and 2000 steps, expect convective cells in temperature and streamfunction plots every `plot_interval` steps.
- **Parallel**: Same physics, split across processes. Output includes console logs and profiling files (plotting optional).

## Key Differences

| Feature                | Serial (`2Dconvection_numba.py`) | Parallel (`2Dconvection_numba_mpi.py`) |
|-----------------------|----------------------------------|----------------------------------------|
| Execution             | Single process                   | Multiple processes (MPI)              |
| Domain                | Full grid on one CPU             | Split along y-axis across ranks       |
| Plotting              | Enabled by default               | Disabled (uncomment for rank 0)       |
| Performance           | Numba JIT                        | Numba JIT + MPI parallelism           |
| Boundary Exchange     | Not applicable                   | MPI Sendrecv for adjacent subgrids    |

## Performance

- **Serial**: Numba accelerates computation significantly over pure Python.
- **Parallel**: Scales with process count, though communication overhead grows with more ranks. Requires even numbers of processes for current boundary exchange logic.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

