#!/usr/bin/env python3
"""
Engineering Example: Signal Processing and Noise Modeling

This example demonstrates how to use BICEP for various engineering applications
including noise generation, signal filtering, and stochastic system modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from bicep_core import BICEPCore, BICEPConfig

def generate_colored_noise():
    """
    Generate different types of noise for signal processing applications.
    Includes white, pink, and brown noise generation.
    """
    
    # Signal parameters
    sampling_rate = 44100  # Hz (standard audio)
    duration = 2.0  # seconds
    n_samples = int(sampling_rate * duration)
    
    config = BICEPConfig(
        device='cpu',
        use_half_precision=False  # Full precision for audio
    )
    bicep = BICEPCore(config)
    
    # Generate base Brownian paths
    n_channels = 3  # White, pink, brown
    paths = bicep.generate_paths(
        n_paths=n_channels,
        n_steps=n_samples-1,
        T=duration,
        control_parameter=0.5,
        time_decay=0.0
    )
    
    # Extract different noise types
    white_noise = np.diff(paths[0, :])  # Derivative of Brownian = white
    brown_noise = paths[1, :]  # Brownian motion = brown noise
    
    # Pink noise (1/f): Apply frequency shaping
    pink_filter = signal.firwin(65, 0.5, window='hamming')
    pink_noise = signal.lfilter(pink_filter, 1.0, white_noise)
    
    # Time array
    t = np.linspace(0, duration, n_samples)
    
    # Frequency analysis
    freqs = np.fft.rfftfreq(n_samples, 1/sampling_rate)
    
    # Compute power spectral density
    white_psd = np.abs(np.fft.rfft(white_noise))**2
    pink_psd = np.abs(np.fft.rfft(pink_noise))**2
    brown_psd = np.abs(np.fft.rfft(brown_noise[:-1]))**2
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Time domain plots
    plt.subplot(2, 3, 1)
    plt.plot(t[:1000], white_noise[:1000], 'b-', linewidth=0.5)
    plt.title('White Noise (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(t[:1000], pink_noise[:1000], 'r-', linewidth=0.5)
    plt.title('Pink Noise (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(t[:1000], brown_noise[:1000], 'brown', linewidth=0.5)
    plt.title('Brown Noise (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Frequency domain plots (log-log)
    plt.subplot(2, 3, 4)
    plt.loglog(freqs[1:], white_psd[1:], 'b-', alpha=0.7)
    plt.title('White Noise PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.loglog(freqs[1:], pink_psd[1:], 'r-', alpha=0.7)
    plt.loglog(freqs[1:], 1/freqs[1:] * np.max(pink_psd) * 0.1, 'k--', label='1/f')
    plt.title('Pink Noise PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.loglog(freqs[1:], brown_psd[1:], 'brown', alpha=0.7)
    plt.loglog(freqs[1:], 1/freqs[1:]**2 * np.max(brown_psd) * 0.01, 'k--', label='1/f²')
    plt.title('Brown Noise PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('colored_noise.png', dpi=300)
    plt.show()
    
    print("Noise Generation Statistics:")
    print(f"White noise RMS: {np.sqrt(np.mean(white_noise**2)):.4f}")
    print(f"Pink noise RMS: {np.sqrt(np.mean(pink_noise**2)):.4f}")
    print(f"Brown noise RMS: {np.sqrt(np.mean(brown_noise**2)):.4f}")
    
    return white_noise, pink_noise, brown_noise

def simulate_vibration_analysis():
    """
    Simulate random vibrations in mechanical systems.
    Models a structure under stochastic loading.
    """
    
    # System parameters
    natural_freq = 50  # Hz
    damping_ratio = 0.05
    mass = 10  # kg
    duration = 10  # seconds
    sampling_rate = 1000  # Hz
    
    config = BICEPConfig(device='cpu')
    bicep = BICEPCore(config)
    
    # Generate random force input
    n_steps = int(duration * sampling_rate)
    force_paths = bicep.generate_paths(
        n_paths=1,
        n_steps=n_steps-1,
        T=duration,
        control_parameter=0.3,  # Some correlation in forcing
        time_decay=0.01
    )
    
    # Extract force signal
    force = np.diff(force_paths[0, :]) * 100  # Scale to realistic force levels (N)
    
    # Create transfer function for the system
    omega_n = 2 * np.pi * natural_freq
    num = [1]
    den = [1/omega_n**2, 2*damping_ratio/omega_n, 1]
    system = signal.TransferFunction(num, den)
    
    # Simulate response
    t = np.linspace(0, duration, n_steps-1)
    _, response, _ = signal.lsim(system, force, t)
    
    # Calculate RMS and peak values
    force_rms = np.sqrt(np.mean(force**2))
    response_rms = np.sqrt(np.mean(response**2))
    peak_response = np.max(np.abs(response))
    
    # Frequency response
    freqs = np.fft.rfftfreq(len(force), 1/sampling_rate)
    force_fft = np.fft.rfft(force)
    response_fft = np.fft.rfft(response)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Time histories
    plt.subplot(2, 2, 1)
    plt.plot(t[:2000], force[:2000], 'b-', linewidth=0.5)
    plt.title('Random Force Input')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(t[:2000], response[:2000], 'r-', linewidth=0.5)
    plt.title('Vibration Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True, alpha=0.3)
    
    # Frequency domain
    plt.subplot(2, 2, 3)
    plt.semilogy(freqs[:500], np.abs(force_fft[:500])**2, 'b-', alpha=0.7)
    plt.axvline(natural_freq, color='k', linestyle='--', label=f'fn = {natural_freq} Hz')
    plt.title('Force Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.semilogy(freqs[:500], np.abs(response_fft[:500])**2, 'r-', alpha=0.7)
    plt.axvline(natural_freq, color='k', linestyle='--', label=f'fn = {natural_freq} Hz')
    plt.title('Response Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vibration_analysis.png', dpi=300)
    plt.show()
    
    print(f"\nVibration Analysis Results:")
    print(f"Force RMS: {force_rms:.2f} N")
    print(f"Response RMS: {response_rms*1000:.2f} mm")
    print(f"Peak displacement: {peak_response*1000:.2f} mm")
    print(f"Dynamic amplification: {response_rms/force_rms*omega_n**2:.2f}")
    
    return force, response, t

def simulate_network_traffic():
    """
    Model network packet arrival times and queue lengths
    using stochastic processes.
    """
    
    # Network parameters
    mean_arrival_rate = 1000  # packets/second
    service_rate = 1200  # packets/second
    simulation_time = 60  # seconds
    time_resolution = 0.001  # 1ms resolution
    
    config = BICEPConfig(
        device='cpu',
        use_memory_pool=True
    )
    bicep = BICEPCore(config)
    
    # Generate arrival process (Poisson-like)
    n_steps = int(simulation_time / time_resolution)
    arrival_paths = bicep.generate_paths(
        n_paths=1,
        n_steps=n_steps-1,
        T=simulation_time,
        control_parameter=0.2,  # Low correlation for Poisson-like behavior
        time_decay=0.0
    )
    
    # Convert to arrival counts
    arrival_increments = np.abs(np.diff(arrival_paths[0, :]))
    arrivals = np.random.poisson(arrival_increments * mean_arrival_rate * time_resolution)
    
    # Simulate queue dynamics
    queue_length = np.zeros(n_steps)
    departures = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        # Add arrivals
        queue_length[i] = queue_length[i-1] + arrivals[i-1]
        
        # Process departures (if queue not empty)
        if queue_length[i] > 0:
            potential_departures = np.random.poisson(service_rate * time_resolution)
            departures[i] = min(potential_departures, queue_length[i])
            queue_length[i] -= departures[i]
    
    # Calculate statistics
    avg_queue_length = np.mean(queue_length)
    max_queue_length = np.max(queue_length)
    utilization = np.mean(queue_length > 0)
    avg_delay = avg_queue_length / mean_arrival_rate  # Little's Law
    
    # Downsample for visualization
    viz_step = 100  # Show every 100ms
    t_viz = np.arange(0, n_steps, viz_step) * time_resolution
    queue_viz = queue_length[::viz_step]
    arrivals_viz = np.sum(arrivals.reshape(-1, viz_step), axis=1)
    departures_viz = np.sum(departures[1:].reshape(-1, viz_step), axis=1)
    
    # Visualization
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t_viz, queue_viz, 'b-', linewidth=1)
    plt.fill_between(t_viz, 0, queue_viz, alpha=0.3)
    plt.title('Queue Length Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Packets')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.bar(t_viz[:100], arrivals_viz[:100], width=0.1, alpha=0.7, label='Arrivals')
    plt.bar(t_viz[:100], -departures_viz[:100], width=0.1, alpha=0.7, label='Departures')
    plt.title('Packet Flow (First 10 seconds)')
    plt.xlabel('Time (s)')
    plt.ylabel('Packets per 100ms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(queue_length, bins=50, density=True, alpha=0.7, color='green')
    plt.title('Queue Length Distribution')
    plt.xlabel('Queue Length')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Autocorrelation of queue length
    lags = np.arange(0, 1000)
    autocorr = [np.corrcoef(queue_length[:-lag], queue_length[lag:])[0, 1] 
                for lag in lags[1:]]
    plt.plot(lags[1:] * time_resolution, autocorr, 'r-', linewidth=2)
    plt.title('Queue Length Autocorrelation')
    plt.xlabel('Lag (s)')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('network_traffic.png', dpi=300)
    plt.show()
    
    print(f"\nNetwork Traffic Statistics:")
    print(f"Average queue length: {avg_queue_length:.1f} packets")
    print(f"Maximum queue length: {max_queue_length:.0f} packets")
    print(f"System utilization: {utilization:.1%}")
    print(f"Average delay (Little's Law): {avg_delay*1000:.1f} ms")
    
    return queue_length, arrivals, departures

if __name__ == "__main__":
    print("BICEP Engineering Example: Signal Processing & Systems")
    print("=" * 60)
    
    # Example 1: Colored noise generation
    print("\n1. Generating colored noise for signal processing:")
    white, pink, brown = generate_colored_noise()
    
    # Example 2: Vibration analysis
    print("\n2. Random vibration analysis:")
    force, response, time = simulate_vibration_analysis()
    
    # Example 3: Network traffic modeling
    print("\n3. Network queue simulation:")
    queue, arr, dep = simulate_network_traffic()
    
    print("\nEngineering simulations complete!")