#!/usr/bin/env python3
"""
Physics Example: Particle Diffusion in a Medium

This example demonstrates how to use BICEP to simulate the Brownian motion
of particles suspended in a fluid, a fundamental problem in statistical physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from bicep_core import BICEPCore, BICEPConfig

def simulate_particle_diffusion():
    """
    Simulate diffusion of particles in water at room temperature.
    
    Physical parameters:
    - Diffusion coefficient for small particles in water: ~10^-9 m²/s
    - Temperature: 293K (20°C)
    - Viscosity of water: 0.001 Pa·s
    """
    
    # Physical constants
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    T = 293  # Temperature (K)
    eta = 0.001  # Viscosity of water (Pa·s)
    radius = 1e-6  # Particle radius (1 μm)
    
    # Einstein relation: D = k_B * T / (6 * π * η * r)
    D = k_B * T / (6 * np.pi * eta * radius)
    print(f"Diffusion coefficient: {D:.2e} m²/s")
    
    # Simulation parameters
    n_particles = 1000
    n_steps = 10000
    total_time = 100.0  # seconds
    dt = total_time / n_steps
    
    # BICEP configuration
    config = BICEPConfig(
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        use_half_precision=False,  # Use full precision for scientific accuracy
        use_memory_pool=True
    )
    
    bicep = BICEPCore(config)
    
    # Generate Brownian paths
    # Note: BICEP generates normalized paths, we need to scale by sqrt(2*D*dt)
    paths = bicep.generate_paths(
        n_paths=n_particles,
        n_steps=n_steps,
        T=total_time,
        control_parameter=0.0,  # No external forces
        time_decay=0.0  # Constant diffusion
    )
    
    # Scale to physical units (meters)
    scale_factor = np.sqrt(2 * D * dt)
    paths_physical = paths * scale_factor
    
    # Analysis
    # Mean Square Displacement (MSD)
    time_points = np.linspace(0, total_time, n_steps + 1)
    msd = np.mean(paths_physical**2, axis=0)
    
    # Theoretical MSD for 1D Brownian motion: MSD = 2*D*t
    theoretical_msd = 2 * D * time_points
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Sample particle trajectories
    plt.subplot(1, 2, 1)
    for i in range(min(10, n_particles)):
        plt.plot(time_points, paths_physical[i, :] * 1e6, alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (μm)')
    plt.title('Sample Particle Trajectories')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean Square Displacement
    plt.subplot(1, 2, 2)
    plt.plot(time_points, msd * 1e12, 'b-', label='Simulated', linewidth=2)
    plt.plot(time_points, theoretical_msd * 1e12, 'r--', label='Theoretical', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (μm²)')
    plt.title('Mean Square Displacement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('particle_diffusion.png', dpi=300)
    plt.show()
    
    # Statistical validation
    final_msd_sim = msd[-1]
    final_msd_theory = theoretical_msd[-1]
    relative_error = abs(final_msd_sim - final_msd_theory) / final_msd_theory
    
    print(f"\nValidation Results:")
    print(f"Final MSD (simulated): {final_msd_sim:.2e} m²")
    print(f"Final MSD (theoretical): {final_msd_theory:.2e} m²")
    print(f"Relative error: {relative_error:.2%}")
    
    return paths_physical, time_points

def simulate_anisotropic_diffusion():
    """
    Simulate particles in an anisotropic medium where diffusion 
    varies with direction (e.g., in liquid crystals or biological membranes).
    """
    
    config = BICEPConfig(device='cpu')
    bicep = BICEPCore(config)
    
    n_particles = 500
    n_steps = 5000
    total_time = 50.0
    
    # Anisotropic diffusion: stronger in early times, weaker later
    # This models particles encountering barriers or constraints
    def variance_modulation(t):
        """Time-dependent diffusion coefficient"""
        return np.exp(-0.05 * t)  # Exponential decrease
    
    paths = bicep.generate_paths(
        n_paths=n_particles,
        n_steps=n_steps,
        T=total_time,
        control_parameter=0.3,  # Slight drift due to flow
        time_decay=0.02  # Time-dependent variance
    )
    
    # Analyze confinement
    time_points = np.linspace(0, total_time, n_steps + 1)
    variance = np.var(paths, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, variance, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Variance')
    plt.title('Anomalous Diffusion: Confinement Effects')
    plt.grid(True, alpha=0.3)
    plt.savefig('anomalous_diffusion.png', dpi=300)
    plt.show()
    
    return paths, time_points

if __name__ == "__main__":
    import torch
    
    print("BICEP Physics Example: Particle Diffusion")
    print("=" * 50)
    
    # Example 1: Standard Brownian motion
    print("\n1. Standard particle diffusion in water:")
    paths1, times1 = simulate_particle_diffusion()
    
    # Example 2: Anomalous diffusion
    print("\n2. Anomalous diffusion with confinement:")
    paths2, times2 = simulate_anisotropic_diffusion()
    
    print("\nPhysics simulation complete!")