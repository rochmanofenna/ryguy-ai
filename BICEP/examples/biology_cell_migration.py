#!/usr/bin/env python3
"""
Biology Example: Cell Migration and Random Walk

This example demonstrates how to use BICEP to model cell migration patterns,
including persistent random walks observed in immune cells, cancer cells,
and other motile cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from bicep_core import BICEPCore, BICEPConfig, StreamingBICEP

def simulate_immune_cell_migration():
    """
    Simulate T-cell migration in lymph node tissue.
    
    Biological parameters:
    - Average speed: 10-15 μm/min
    - Persistence time: 2-5 minutes
    - Observation period: 2 hours
    """
    
    # Biological parameters
    cell_speed = 12.5  # μm/min (average T-cell speed)
    persistence_time = 3.5  # minutes
    observation_time = 120  # minutes (2 hours)
    time_step = 0.5  # minutes
    n_steps = int(observation_time / time_step)
    
    # Convert persistence to control parameter
    # Higher persistence = more directed movement
    persistence_param = 1.0 - np.exp(-time_step / persistence_time)
    
    # BICEP configuration
    config = BICEPConfig(
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        use_memory_pool=True
    )
    
    bicep = BICEPCore(config)
    
    # Simulate multiple cells
    n_cells = 100
    
    # Generate 2D trajectories (x and y coordinates)
    x_paths = bicep.generate_paths(
        n_paths=n_cells,
        n_steps=n_steps,
        T=observation_time,
        control_parameter=0.8,  # High persistence
        time_decay=0.01  # Slight decay in directionality over time
    )
    
    y_paths = bicep.generate_paths(
        n_paths=n_cells,
        n_steps=n_steps,
        T=observation_time,
        control_parameter=0.8,
        time_decay=0.01
    )
    
    # Scale to biological units
    x_positions = x_paths * cell_speed * np.sqrt(time_step)
    y_positions = y_paths * cell_speed * np.sqrt(time_step)
    
    # Calculate metrics
    time_points = np.linspace(0, observation_time, n_steps + 1)
    
    # Mean displacement
    displacements = np.sqrt(x_positions**2 + y_positions**2)
    mean_displacement = np.mean(displacements, axis=0)
    
    # Instantaneous velocities
    velocities = np.sqrt(np.diff(x_positions)**2 + np.diff(y_positions)**2) / time_step
    mean_velocity = np.mean(velocities)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Cell trajectories
    plt.subplot(1, 3, 1)
    for i in range(min(20, n_cells)):
        plt.plot(x_positions[i, :], y_positions[i, :], alpha=0.5, linewidth=1)
        plt.plot(x_positions[i, 0], y_positions[i, 0], 'go', markersize=6)  # Start
        plt.plot(x_positions[i, -1], y_positions[i, -1], 'ro', markersize=6)  # End
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')
    plt.title('T-Cell Migration Trajectories')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean displacement over time
    plt.subplot(1, 3, 2)
    plt.plot(time_points, mean_displacement, 'b-', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Mean Displacement (μm)')
    plt.title('Cell Displacement Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Velocity distribution
    plt.subplot(1, 3, 3)
    all_velocities = velocities.flatten()
    plt.hist(all_velocities, bins=50, density=True, alpha=0.7, color='green')
    plt.axvline(mean_velocity, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_velocity:.1f} μm/min')
    plt.xlabel('Velocity (μm/min)')
    plt.ylabel('Probability Density')
    plt.title('Velocity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cell_migration.png', dpi=300)
    plt.show()
    
    print(f"\nMigration Statistics:")
    print(f"Mean velocity: {mean_velocity:.1f} μm/min")
    print(f"Total displacement: {mean_displacement[-1]:.1f} μm")
    print(f"Effective diffusion coefficient: {mean_displacement[-1]**2 / (4 * observation_time):.1f} μm²/min")
    
    return x_positions, y_positions, time_points

def simulate_chemotaxis():
    """
    Simulate directed cell migration in response to chemical gradients (chemotaxis).
    Models neutrophils moving toward infection site.
    """
    
    # Use streaming for large-scale simulation
    config = BICEPConfig(device='cpu')
    streaming_bicep = StreamingBICEP(config, buffer_size=1000)
    
    n_cells = 5000  # Large population
    n_steps = 1000
    total_time = 60  # minutes
    
    # Chemotactic gradient parameters
    gradient_strength = 0.7  # Strong directional bias
    
    # Process cells in batches
    all_endpoints = []
    
    print("\nSimulating chemotaxis for 5000 cells...")
    for i, batch in enumerate(streaming_bicep.stream_generate(
        total_paths=n_cells,
        n_steps=n_steps,
        T=total_time,
        control_parameter=gradient_strength,
        time_decay=0.0  # Constant gradient
    )):
        # Extract final positions
        endpoints = batch[:, -1]
        all_endpoints.extend(endpoints.cpu().numpy())
        
        if i % 5 == 0:
            print(f"  Processed {len(all_endpoints)} cells...")
    
    all_endpoints = np.array(all_endpoints)
    
    # Analyze directional bias
    mean_final_position = np.mean(all_endpoints)
    std_final_position = np.std(all_endpoints)
    
    # Chemotactic index (CI)
    ci = mean_final_position / np.mean(np.abs(all_endpoints))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_endpoints, bins=50, density=True, alpha=0.7, color='blue')
    plt.axvline(mean_final_position, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_final_position:.2f}')
    plt.xlabel('Final Position (relative units)')
    plt.ylabel('Probability Density')
    plt.title('Chemotactic Response Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Show theoretical comparison
    x = np.linspace(-3, 3, 100)
    random_walk = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x, random_walk, 'k--', label='Random Walk', linewidth=2)
    
    # Normalize and plot actual distribution
    hist, bins = np.histogram(all_endpoints / std_final_position, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, hist, 'b-', label='Chemotactic Walk', linewidth=2)
    
    plt.xlabel('Normalized Position')
    plt.ylabel('Probability Density')
    plt.title('Chemotaxis vs Random Walk')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chemotaxis.png', dpi=300)
    plt.show()
    
    print(f"\nChemotaxis Results:")
    print(f"Chemotactic Index: {ci:.3f} (>0 indicates directional bias)")
    print(f"Mean displacement: {mean_final_position:.2f}")
    print(f"Standard deviation: {std_final_position:.2f}")

def simulate_tumor_cell_invasion():
    """
    Model tumor cell invasion with varying microenvironment conditions.
    Demonstrates time-varying control parameters.
    """
    
    config = BICEPConfig(device='cpu', use_half_precision=False)
    bicep = BICEPCore(config)
    
    # Invasion parameters
    n_cells = 200
    n_steps = 2000
    observation_hours = 24
    
    # Time-varying invasion rate (circadian rhythm effects)
    def invasion_rate(t):
        """Sinusoidal variation modeling circadian effects"""
        base_rate = 1.0
        amplitude = 0.3
        period = 12  # hours
        return base_rate + amplitude * np.sin(2 * np.pi * t / period)
    
    # Generate invasion paths with time-varying parameters
    paths = bicep.generate_paths(
        n_paths=n_cells,
        n_steps=n_steps,
        T=observation_hours,
        control_parameter=0.6,  # Moderate persistence
        time_decay=0.05  # Increasing tissue resistance
    )
    
    # Apply time-varying invasion rate
    time_points = np.linspace(0, observation_hours, n_steps + 1)
    invasion_scaling = np.array([invasion_rate(t) for t in time_points])
    
    # Scale paths by invasion rate
    scaled_paths = paths * invasion_scaling[np.newaxis, :]
    
    # Calculate invasion depth
    invasion_depths = np.max(np.abs(scaled_paths), axis=1)
    mean_invasion = np.mean(invasion_depths)
    
    print(f"\nTumor Invasion Statistics:")
    print(f"Mean invasion depth: {mean_invasion:.2f} units")
    print(f"Max invasion depth: {np.max(invasion_depths):.2f} units")
    print(f"Cells invading >50 units: {np.sum(invasion_depths > 50)}/{n_cells}")
    
    return scaled_paths, time_points

if __name__ == "__main__":
    import torch
    
    print("BICEP Biology Example: Cell Migration")
    print("=" * 50)
    
    # Example 1: Immune cell migration
    print("\n1. T-cell migration in lymph node:")
    x_pos, y_pos, times = simulate_immune_cell_migration()
    
    # Example 2: Chemotaxis
    print("\n2. Neutrophil chemotaxis:")
    simulate_chemotaxis()
    
    # Example 3: Tumor invasion
    print("\n3. Tumor cell invasion:")
    tumor_paths, tumor_times = simulate_tumor_cell_invasion()
    
    print("\nBiology simulations complete!")