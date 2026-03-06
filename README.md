AetherKernel — Cosmic Web Simulation Engine

v9/v10 · Physics-Corrected · JAX-Accelerated · Multi-Grid PM · Full-Residual PINN Surrogate

AetherKernel is a research-grade cosmological N-body simulation engine that models the large-scale structure of the universe (the "cosmic web") from first principles. Starting from Gaussian random initial conditions at high redshift, it evolves dark matter particles forward in time under gravity to produce the filaments, voids, sheets, and halos that constitute the observable universe at z=0.

Table of Contents

Features

Requirements

Installation

Architecture

Configuration

Deployment

Outputs

Known Issues

Limitations

Suggested Enhancements

Features

ΛCDM Cosmology — Pre-tabulated growth factors, Eisenstein-Hu transfer function, σ₈ normalization

2LPT Initial Conditions — Second-order Lagrangian Perturbation Theory for accurate particle displacements

Spectral Poisson Solver — FFT-based gravitational potential computation (periodic boundary conditions)

Multi-Grid V-Cycle — Jacobi relaxation with restriction/prolongation for multi-scale Poisson solving

Kick-Drift-Kick (KDK) Leapfrog — Symplectic integrator with Hubble drag for expanding background

Cloud-in-Cell (CIC) PM — Mass deposition and force interpolation, JAX/GPU-portable

Friends-of-Friends Halo Finder — KDTree-based FoF with configurable linking length

Physics-Informed Neural Network (PINN) — PyTorch surrogate trained on Poisson + continuity + Euler residuals

Symbolic Verification — SymPy Friedmann equation solver for axiomatic grounding

Visualization — 2D density slices, power spectrum evolution, PINN predictions, MP4 animation

Requirements

Python Version

Python >= 3.9 

Core Dependencies

PackageVersionPurposenumpy>= 1.23Array operations, FFT, random fieldsscipy>= 1.9FFT, integration, interpolation, spatial KDTreetorch>= 2.0PINN training, autogradjax>= 0.4JIT-compiled spectral solvers, CIC depositionjaxlib>= 0.4JAX backend (CPU or GPU)sympy>= 1.11Symbolic Friedmann equation solvermatplotlib>= 3.6Visualization, animationnetworkx>= 3.0Friends-of-Friends halo graph connectivitynumba>= 0.57JIT acceleration for NumPy-level operations 

Optional Dependencies

PackagePurposeffmpegMP4 animation export (must be in system PATH)cuda / rocmGPU acceleration via JAX and PyTorchcupyGPU-accelerated NumPy arrays (optional enhancement) 

Hardware Requirements

ModeMinimumRecommendedCPU (Ngrid=32)4 GB RAM8 GB RAMCPU (Ngrid=64)8 GB RAM16 GB RAMCPU (Ngrid=128)32 GB RAM64 GB RAMGPU (Ngrid=64)6 GB VRAM12 GB VRAM 

Note: PINN training at n_col=5000, epochs=800 across all snapshots is the dominant memory and compute cost. Reduce n_col or epochs for resource-constrained environments.

Installation

1. Clone and create environment

git clone https://github.com/your-org/aetherkernel.git cd aetherkernel python -m venv .venv source .venv/bin/activate # Windows: .venv\Scripts\activate 

2. Install dependencies

pip install numpy scipy torch sympy matplotlib networkx numba 

For CPU-only JAX:

pip install jax jaxlib 

For GPU (CUDA 12):

pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html pip install torch --index-url https://download.pytorch.org/whl/cu121 

3. Install ffmpeg (for animation export)

# Ubuntu/Debian sudo apt install ffmpeg # macOS brew install ffmpeg # Conda conda install -c conda-forge ffmpeg 

4. Verify installation

python -c " import jax, torch, numpy, scipy, sympy, networkx, numba print('JAX devices:', jax.devices()) print('PyTorch CUDA:', torch.cuda.is_available()) print('All imports OK') " 

Architecture

AetherKernel is implemented as a single class AetherCosmicWebKernel_v9 with several well-separated subsystems:

AetherCosmicWebKernel_v9 │ ├── Cosmological Layer │ ├── H(a) — Hubble parameter E(a) │ ├── growth_factor(a) — D₁(a) via tabulated integral │ ├── second_growth_factor(a) — D₂(a) approximation │ ├── transfer(k) — Eisenstein-Hu T(k) │ └── compute_sigma8() — σ₈ normalization integral │ ├── Initial Conditions │ ├── generate_gaussian_random_field() — P(k)-seeded δ(x) │ └── compute_2lpt() — 2nd-order Lagrangian displacements │ ├── Gravity Solvers (JAX) │ ├── _compute_potential_jax() — Spectral Poisson solver │ ├── _compute_forces_jax() — Spectral force kernel │ ├── _relax_jacobi() — Jacobi smoother │ ├── _restrict() / _prolong() — Multi-grid operators │ ├── _v_cycle() — V-cycle recursion │ └── _multi_grid_poisson_jax() — Multi-grid driver │ ├── Particle-Mesh (JAX) │ ├── _deposit_cic_jax() — CIC mass deposition (scatter_add) │ └── _interpolate_cic_jax() — CIC force interpolation (vmap) │ ├── Integrator │ └── leapfrog_kdk() — KDK with Hubble drag │ ├── Analysis │ ├── compute_power_spectrum() — Binned 3D P(k) │ ├── linear_theory_pk() — Theoretical P(k) at z=0 │ └── find_halos() — FoF halo finder (KDTree + NetworkX) │ ├── PINN Surrogate (PyTorch) │ ├── CosmoPINN — 4-layer MLP: (x,y,z,a) → (δ, vx, vy, vz, φ) │ ├── train_pinn_surrogate() — Data + physics residual training │ └── predict_pinn() — Inference on full grid │ └── Symbolic / Demo ├── symbolic_friedmann() — SymPy Friedmann ODE solver └── run_demo() — End-to-end pipeline runner 

Data Flow

Cosmological Params │ ▼ Gaussian Random Field (P(k), T(k), D₁(a_init)) │ ▼ 2LPT Initial Conditions (φ₁, φ₂ → particle displacements) │ ▼ Particle Positions + Velocities │ ▼ ┌──────────────────────────────┐ │ KDK Leapfrog Loop (N steps) │ │ ┌────────────────────────┐ │ │ │ Kick ½ (old forces) │ │ │ │ Drift (update pos) │ │ │ │ CIC deposit → δ_new │ │ │ │ Kick ½ (new forces) │ │ │ └────────────────────────┘ │ └──────────────────────────────┘ │ ▼ Snapshot History (δ, a) │ ┌───┴─────────────────┐ ▼ ▼ Power Spectrum PINN Surrogate Analysis Training + Inference │ │ └──────────┬───────────┘ ▼ Visualization (slices, P(k), animation) 

Backend Strategy

ComponentBackendReasonGrowth factor tabulationNumPy + SciPyOne-time cost; readabilityPoisson / force solverJAX (@jit)Hot loop; JIT + GPU-portableCIC depositionJAX (lax.scatter_add)Differentiable; GPU-portableCIC interpolationJAX (vmap)Vectorized; GPU-portablePINN trainingPyTorchMature autograd ecosystemSymbolic verificationSymPyExact symbolic mathHalo findingSciPy KDTree + NetworkXEfficient spatial indexing 

Configuration

All parameters are passed to the constructor:

kernel = AetherCosmicWebKernel_v9( Ngrid = 64, # Grid resolution (particles = Ngrid³) Lbox = 200.0, # Box size in Mpc/h Omega_m = 0.3, # Matter density parameter Omega_L = 0.7, # Dark energy density parameter Omega_b = 0.045, # Baryon density parameter h = 0.7, # Dimensionless Hubble constant seed = 137, # Random seed for reproducibility sigma8 = 0.8, # Amplitude of matter fluctuations ns = 0.96, # Spectral index of primordial power spectrum ) 

run_demo Parameters

kernel.run_demo( n_steps = 100, # Number of timesteps (log-spaced in scale factor) a_init = 0.1, # Initial scale factor (z=9) a_final = 1.0, # Final scale factor (z=0) ) 

PINN Training Parameters

kernel.train_pinn_surrogate( delta_history, a_history, epochs = 800, lr = 0.003, lambda_phys = 1.0, # Weight of physics residual loss n_col = 5000, # Collocation points per epoch ) 

Deployment

Local Development

from aetherkernel import AetherCosmicWebKernel_v9 kernel = AetherCosmicWebKernel_v9(Ngrid=64, Lbox=200.0, seed=137) kernel.symbolic_friedmann() final_delta = kernel.run_demo(n_steps=100) 

High-Resolution Run (HPC Cluster)

For Ngrid >= 128, use a SLURM job script:

#!/bin/bash #SBATCH --job-name=aetherkernel #SBATCH --nodes=1 #SBATCH --ntasks=1 #SBATCH --cpus-per-task=16 #SBATCH --mem=128G #SBATCH --gres=gpu:1 #SBATCH --time=12:00:00 module load python/3.11 cuda/12.0 source .venv/bin/activate python -c " from aetherkernel import AetherCosmicWebKernel_v9 kernel = AetherCosmicWebKernel_v9(Ngrid=256, Lbox=500.0, seed=42) kernel.run_demo(n_steps=200, a_init=0.02, a_final=1.0) " 

GPU Activation

JAX will automatically use a detected GPU. To force a specific device:

import jax # Force CPU jax.config.update("jax_platform_name", "cpu") # Check active devices print(jax.devices()) 

For PyTorch (PINN training):

import torch device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Note: current PINN code runs on CPU — moving to GPU requires adding .to(device) calls 

Docker

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg RUN pip install numpy scipy sympy matplotlib networkx numba \ torch --index-url https://download.pytorch.org/whl/cu121 \ jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html WORKDIR /app COPY . . CMD ["python", "aetherkernel.py"] 

Outputs

FileDescriptionaetherkernel_v9_cosmic_web.png4-panel figure: initial slice, final slice, P(k) evolution, PINN predictioncosmic_web_3d_evolution_v9.mp4Animated density evolution (requires ffmpeg)stdoutStep-by-step diagnostics: σ(δ), mass conservation error, PINN epoch loss 

Known Issues

1. Multi-Grid V-Cycle Cannot Be Fully JIT-Compiled

Location: _v_cycle() / _multi_grid_poisson_jax()

_v_cycle is a Python-level recursive function. JAX's @jit traces the call graph at compile time but cannot unroll dynamic Python recursion. As currently written, JAX will either raise a tracing error or silently fall back to Python-level execution, negating the JIT benefit.

Workaround: The spectral solver _compute_potential_jax remains JIT-compiled and is used as the coarse-grid solver at the recursion base case. For most use cases this is accurate enough.

Fix: Replace Python recursion with a fixed-depth unrolled loop, or use jax.lax.while_loop for a fully traceable implementation.

2. itertools.product Inside a JIT-Traced Function

Location: _deposit_cic_jax()

itertools.product is called at Python trace time to generate the 8 CIC offset corners. This works because the offsets are constant and get baked into the traced computation, but it is fragile — a JAX version upgrade or stricter tracing mode could break it.

Fix: Replace with a static jnp.array of shape (8, 3) defined at class initialization.

3. _prolong Boundary Wrapping

Location: _prolong()

The prolongation operator interpolates edge values using fine[2::2], which at the grid boundary references indices that may be out-of-bounds or wrap incorrectly for a periodic box.

Fix: Apply jnp.roll-based periodic wrapping explicitly at boundary indices.

4. Second-Order Growth Factor Approximation

Location: _second_growth_factor_raw()

The approximation D₂ ≈ -3/7 D₁² Ω_m(a)^(-1/55) uses an unusual exponent. The standard fitting formula in the literature uses Ω_m(a)^(1/143) or the exact EdS result D₂ = -3/7 D₁². The -1/55 exponent may be a transcription error or a non-standard variant, and should be verified against a reference (e.g., Crocce & Scoccimarro 2006).

5. PINN Not Moved to GPU

Location: train_pinn_surrogate() / CosmoPINN

PyTorch model and tensors are created on CPU. For Ngrid=64 with epochs=800 and n_col=5000, training is very slow on CPU (potentially hours).

Fix: Add .to(device) calls for model, inputs, and targets, and expose a device argument.

6. Class Name / Version Mismatch

The class is named AetherCosmicWebKernel_v9, the print statement says v9, but the file is labelled v10. This inconsistency should be resolved to avoid confusion in logs and downstream usage.

7. JAX PRNG Key Unused

Location: __init__

jax.random.PRNGKey(seed) 

The returned key is discarded. All random field generation uses np.random.seed(seed) instead. JAX's random functions require the key to be explicitly threaded through calls — simply creating it and discarding it provides no reproducibility guarantee for JAX operations.

8. Power Spectrum Loop Performance

Location: compute_power_spectrum()

The binning loop iterates over nbins with a boolean mask at each bin, giving O(nbins × Ngrid³) complexity. For Ngrid=128, this is noticeably slow.

Fix: Use np.digitize or np.histogram for O(Ngrid³ log Ngrid) binning.

Limitations

Physical

No baryonic physics — Gas dynamics, star formation, AGN feedback, and radiative cooling are absent. This is a dark-matter-only (DMO) simulation.

Newtonian gravity only — General relativistic corrections (relevant at horizon scales, k ~ H/c) are not included.

Flat ΛCDM only — Curvature (Ω_k ≠ 0), dynamical dark energy (w ≠ -1), and massive neutrinos are not supported.

Single species — All particles have equal mass. There is no distinction between cold dark matter and baryons in the dynamics.

PM (not Tree-PM) — The Particle-Mesh method lacks a short-range tree correction, meaning force resolution is limited to ~2 grid cells. Sub-grid clustering is suppressed.

Softening length — Set to 1.5 * dx, which is relatively large. Dense halo cores will be over-softened.

Numerical

Grid resolution — At Ngrid=64, the simulation resolves large-scale structure reasonably but misses small-scale power. Production cosmological simulations use Ngrid ≥ 512–4096.

CIC aliasing — Cloud-in-Cell deposition introduces aliasing at the Nyquist frequency. A dealiasing correction (interlacing or compensation kernel) is not applied.

Fixed timestep spacing — Timesteps are log-spaced in a, which is a reasonable default but not adaptive. During rapid structure formation, a smaller time step may be needed.

PINN Surrogate

Small architecture — The 3-layer, 256-unit MLP is likely underpowered to represent a 64³ density field across all redshifts simultaneously. It will underfit at small scales.

No uncertainty quantification — The surrogate produces point predictions with no error estimate.

Training on simulation snapshots only — The PINN learns from a single simulation trajectory (single seed, single cosmology). It will not generalize across cosmological parameters without retraining.

Slow training — 800 epochs over all snapshots with 5000 collocation points each is O(hours) on CPU.

Suggested Enhancements

Near-Term (Low Effort, High Impact)

1. Fix the V-Cycle JIT Issue Replace Python recursion with a statically-unrolled loop over a fixed number of levels. This allows _multi_grid_poisson_jax to be fully JIT-compiled.

def _v_cycle_unrolled(self, u, rhs, n_levels=4): # Descend residuals, us = [], [] for _ in range(n_levels): for _ in range(5): u = self._relax_jacobi(u, rhs) res = self._compute_residual(u, rhs) residuals.append(rhs); us.append(u) rhs = self._restrict(res); u = jnp.zeros_like(rhs) # Coarse solve u = self._compute_potential_jax(rhs, 1.0) # Ascend for rhs_old, u_old in zip(reversed(residuals), reversed(us)): u = u_old + self._prolong(u) for _ in range(5): u = self._relax_jacobi(u, rhs_old) return u 

2. Vectorized Power Spectrum Binning

def compute_power_spectrum(self, delta, nbins=40): delta_k = fftn(delta) k = np.sqrt(self.k2_mesh).flatten() pk = (np.abs(delta_k)**2 / self.Ngrid**6).flatten() * self.Lbox**3 kbins = np.logspace(np.log10(0.01), np.log10(np.pi / self.dx), nbins + 1) bin_idx = np.digitize(k, kbins) - 1 Pk = np.bincount(bin_idx, weights=pk, minlength=nbins) counts = np.bincount(bin_idx, minlength=nbins) mask = counts > 0 return (kbins[:-1] + kbins[1:])[mask] / 2, (Pk / counts)[mask] 

3. Thread the JAX PRNG Key

self.rng_key = jax.random.PRNGKey(seed) # Use: self.rng_key, subkey = jax.random.split(self.rng_key) 

4. Move PINN to GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") model = self.CosmoPINN().to(device) # ... move all tensors with .to(device) 

Medium-Term (Moderate Effort)

5. Tree-PM Short-Range Force Add a direct particle-particle or tree force correction inside a softening radius to improve sub-grid force accuracy. This upgrades PM → TreePM, enabling better halo density profiles.

6. Adaptive Timestepping Replace log-spaced a steps with a timestep criterion based on the maximum force magnitude or the Courant condition: dt = α * dx / max(|v|).

7. Cosmological Parameter Emulation with PINN Extend the PINN inputs to include (x, y, z, a, Ω_m, σ₈), and train across a Latin hypercube of cosmologies. This transforms the surrogate into a proper cosmological emulator.

8. Halo Mass Function Output After find_halos(), compute and plot the differential halo mass function dn/d ln M and compare to analytic Press-Schechter or Tinker fits. This is a standard validation diagnostic.

9. 3D Visualization Replace 2D slice plots with a volumetric renderer using ipyvolume, yt, or k3d for Jupyter environments.

Long-Term (Significant Effort)

10. Multi-Cosmology Training Pipeline Build an automated pipeline to: (a) run a suite of simulations across a cosmological parameter grid, (b) train a single PINN surrogate on all outputs, and (c) use it for fast posterior inference (simulation-based inference / likelihood-free inference).

11. Baryon Physics (Hydro) Couple the PM gravity solver to a Smoothed Particle Hydrodynamics (SPH) or grid-based Eulerian hydro solver to include gas, star formation, and feedback.

12. MPI Parallelism Decompose the domain into slabs for distributed-memory parallelism using mpi4py + parallel FFT (pfft or mpi4jax), enabling Ngrid >= 512 on multi-node clusters.

13. Differentiable Simulation (End-to-End) Port the entire PM loop to JAX (including CIC and leapfrog) to make the simulation end-to-end differentiable. This enables gradient-based inference of cosmological parameters directly from observed density fields using tools like jax-cosmo.

Citation

If you use AetherKernel in your research, please cite the relevant methodological references:

Eisenstein & Hu (1998) — Transfer function

Crocce & Scoccimarro (2006) — 2LPT

Davis et al. (1985) — Friends-of-Friends halo finder

Raissi et al. (2019) — Physics-Informed Neural Networks

License

See LICENSE for details.

