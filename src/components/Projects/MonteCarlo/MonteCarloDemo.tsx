'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';
import { MonteCarloWebGPU, MonteCarloCPU, OptionParams, PricingResult } from './MonteCarloWebGPU';

// Dynamically import heavy components
const Plot = dynamic(() => import('react-plotly.js').then((mod) => mod.default), {
  ssr: false,
  loading: () => <div className="h-64 bg-slate-800/50 animate-pulse rounded" />
});
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

interface GreekValues {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export default function MonteCarloDemo() {
  // Option parameters
  const [params, setParams] = useState<OptionParams>({
    S: 100,      // Current stock price
    K: 100,      // Strike price
    r: 0.05,     // Risk-free rate (5%)
    T: 1,        // 1 year to maturity
    sigma: 0.2,  // 20% volatility
    paths: 100000,
    steps: 252   // Daily steps for 1 year
  });

  // Results and performance
  const [results, setResults] = useState<PricingResult | null>(null);
  const [greeks, setGreeks] = useState<GreekValues | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState<'auto' | 'webgpu' | 'cpu'>('auto');
  const [showCode, setShowCode] = useState(false);
  const [showVolatilitySurface, setShowVolatilitySurface] = useState(false);
  const [performanceHistory, setPerformanceHistory] = useState<PricingResult[]>([]);

  // Engine instances
  const webgpuEngine = useRef<MonteCarloWebGPU | null>(null);
  const cpuEngine = useRef<MonteCarloCPU | null>(null);

  // Initialize engines
  useEffect(() => {
    const init = async () => {
      webgpuEngine.current = new MonteCarloWebGPU();
      const webgpuSupported = await webgpuEngine.current.initialize();

      cpuEngine.current = new MonteCarloCPU();

      if (!webgpuSupported && selectedDevice === 'webgpu') {
        setSelectedDevice('cpu');
        console.warn('WebGPU not supported, falling back to CPU');
      }
    };

    init();

    return () => {
      if (webgpuEngine.current) {
        webgpuEngine.current.destroy();
      }
    };
  }, []);

  // Calculate option price
  const calculatePrice = useCallback(async () => {
    setIsCalculating(true);

    try {
      let result: PricingResult;

      if (selectedDevice === 'webgpu' && webgpuEngine.current) {
        result = await webgpuEngine.current.priceOption(params);
      } else if (cpuEngine.current) {
        result = await cpuEngine.current.priceOption(params);
      } else {
        throw new Error('No pricing engine available');
      }

      setResults(result);
      setPerformanceHistory(prev => [...prev.slice(-19), result]);

      // Calculate Greeks in background
      if (selectedDevice === 'webgpu' && webgpuEngine.current) {
        const greekValues = await webgpuEngine.current.calculateGreeks(params);
        setGreeks(greekValues);
      }
    } catch (error) {
      console.error('Pricing calculation failed:', error);
    } finally {
      setIsCalculating(false);
    }
  }, [params, selectedDevice]);

  // Auto-calculate on parameter change (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isCalculating) {
        calculatePrice();
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [params]);

  // Generate volatility surface data
  const generateVolatilitySurface = () => {
    const strikes = Array.from({ length: 20 }, (_, i) => 80 + i * 2);
    const maturities = Array.from({ length: 20 }, (_, i) => 0.25 + i * 0.25);
    const surface: number[][] = [];

    for (const T of maturities) {
      const row: number[] = [];
      for (const K of strikes) {
        // Simplified implied volatility calculation
        const moneyness = K / params.S;
        const timeEffect = Math.sqrt(T);
        const impliedVol = params.sigma * (0.8 + 0.4 * Math.abs(1 - moneyness) * timeEffect);
        row.push(impliedVol);
      }
      surface.push(row);
    }

    return { strikes, maturities, surface };
  };

  const shaderCode = `
// WebGPU Compute Shader for Monte Carlo Option Pricing
@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;

struct Params {
  S: f32,         // Current stock price
  K: f32,         // Strike price
  r: f32,         // Risk-free rate
  T: f32,         // Time to maturity
  sigma: f32,     // Volatility
  paths: u32,     // Number of paths
  steps: u32,     // Steps per path
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= params.paths) { return; }

  var price = params.S;
  let dt = params.T / f32(params.steps);
  let drift = (params.r - 0.5 * params.sigma * params.sigma) * dt;
  let diffusion = params.sigma * sqrt(dt);

  // Simulate Geometric Brownian Motion path
  for (var i = 0u; i < params.steps; i++) {
    let z = generateNormal(idx, i);  // Box-Muller transform
    price *= exp(drift + diffusion * z);
  }

  // Calculate and store discounted payoff
  let payoff = max(price - params.K, 0.0);
  results[idx] = payoff * exp(-params.r * params.T);
}`;

  const cudaCode = `
// CUDA Kernel Reference (for comparison - not executed in browser)
// Native CUDA would run 50-100x+ faster than JavaScript
__global__ void monteCarloKernel(
  float* results,
  float S, float K, float r, float T, float sigma,
  int paths, int steps
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths) return;

  curandState state;
  curand_init(clock64(), idx, 0, &state);

  float price = S;
  float dt = T / steps;
  float drift = (r - 0.5f * sigma * sigma) * dt;
  float diffusion = sigma * sqrtf(dt);

  // Simulate path using Geometric Brownian Motion
  for (int i = 0; i < steps; i++) {
    float z = curand_normal(&state);
    price *= expf(drift + diffusion * z);
  }

  // Store discounted payoff
  float payoff = fmaxf(price - K, 0.0f);
  results[idx] = payoff * expf(-r * T);
}`;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Monte Carlo Option Pricing
        </h1>
        <p className="text-gray-400">WebGPU-Accelerated Black-Scholes Monte Carlo Simulation</p>
        <div className="flex gap-4 mt-4 text-sm">
          <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded">
            {results?.device === 'webgpu' ? 'WebGPU Active' : 'CPU Mode'}
          </span>
          <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded">
            {results ? `${(results.pathsPerSecond / 1e6).toFixed(2)}M paths/sec` : 'Ready'}
          </span>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Panel */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700"
        >
          <h2 className="text-xl font-semibold mb-4">Parameters</h2>

          <div className="space-y-4">
            {/* Spot Price */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Spot Price (S)</span>
                <span className="text-cyan-400">${params.S.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="50"
                max="150"
                step="1"
                value={params.S}
                onChange={(e) => setParams({ ...params, S: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Strike Price */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Strike Price (K)</span>
                <span className="text-cyan-400">${params.K.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="50"
                max="150"
                step="1"
                value={params.K}
                onChange={(e) => setParams({ ...params, K: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Volatility */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Volatility (σ)</span>
                <span className="text-cyan-400">{(params.sigma * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.01"
                value={params.sigma}
                onChange={(e) => setParams({ ...params, sigma: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Risk-free Rate */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Risk-free Rate (r)</span>
                <span className="text-cyan-400">{(params.r * 100).toFixed(1)}%</span>
              </label>
              <input
                type="range"
                min="0"
                max="0.1"
                step="0.001"
                value={params.r}
                onChange={(e) => setParams({ ...params, r: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Time to Maturity */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Time to Maturity (T)</span>
                <span className="text-cyan-400">{params.T.toFixed(2)} years</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={params.T}
                onChange={(e) => setParams({ ...params, T: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Monte Carlo Paths */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>MC Paths</span>
                <span className="text-cyan-400">{(params.paths / 1000).toFixed(0)}K</span>
              </label>
              <input
                type="range"
                min="10000"
                max="2000000"
                step="10000"
                value={params.paths}
                onChange={(e) => setParams({ ...params, paths: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>
          </div>

          {/* Device Selection */}
          <div className="mt-6 space-y-2">
            <label className="text-sm text-gray-400">Compute Device</label>
            <div className="grid grid-cols-3 gap-2">
              {(['auto', 'webgpu', 'cpu'] as const).map((device) => (
                <button
                  key={device}
                  onClick={() => setSelectedDevice(device)}
                  className={`px-3 py-2 rounded text-sm transition-all ${
                    selectedDevice === device
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-700 text-gray-400 hover:bg-slate-600'
                  }`}
                >
                  {device.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Results Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:col-span-2 space-y-6"
        >
          {/* Pricing Results */}
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold mb-4">Pricing Results</h2>

            {results ? (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Option Price</p>
                  <p className="text-2xl font-bold text-cyan-400">
                    ${results.price.toFixed(4)}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    ±{results.standardError.toFixed(4)}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-400">95% Confidence</p>
                  <p className="text-lg text-gray-300">
                    [{results.confidence95[0].toFixed(3)}, {results.confidence95[1].toFixed(3)}]
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-400">Execution Time</p>
                  <p className="text-2xl font-bold text-green-400">
                    {results.executionTime.toFixed(0)}ms
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-400">Moneyness</p>
                  <p className="text-lg text-gray-300">
                    {(params.S / params.K).toFixed(3)}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-400">Compute Device</p>
                  <p className="text-lg text-blue-400 uppercase">
                    {results.device}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-400">Throughput</p>
                  <p className="text-lg text-purple-400">
                    {(results.pathsPerSecond / 1e6).toFixed(2)}M/s
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-32">
                <div className="text-gray-500">
                  {isCalculating ? 'Calculating...' : 'Waiting for calculation'}
                </div>
              </div>
            )}
          </div>

          {/* Greeks */}
          {greeks && (
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4">Greeks</h2>
              <div className="grid grid-cols-5 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Δ Delta</p>
                  <p className="text-xl font-semibold">{greeks.delta.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Γ Gamma</p>
                  <p className="text-xl font-semibold">{greeks.gamma.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Θ Theta</p>
                  <p className="text-xl font-semibold">{greeks.theta.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">ν Vega</p>
                  <p className="text-xl font-semibold">{greeks.vega.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">ρ Rho</p>
                  <p className="text-xl font-semibold">{greeks.rho.toFixed(4)}</p>
                </div>
              </div>
            </div>
          )}

          {/* Performance History */}
          {performanceHistory.length > 0 && (
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4">Performance History</h2>
              <Plot
                data={[
                  {
                    x: performanceHistory.map((_, i) => i),
                    y: performanceHistory.map(r => r.executionTime),
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: '#06b6d4' },
                    name: 'Execution Time (ms)'
                  }
                ]}
                layout={{
                  height: 200,
                  margin: { t: 10, b: 30, l: 40, r: 10 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#9ca3af' },
                  xaxis: { gridcolor: '#374151' },
                  yaxis: { gridcolor: '#374151', title: { text: 'Time (ms)' } }
                }}
                config={{ displayModeBar: false }}
              />
            </div>
          )}
        </motion.div>
      </div>

      {/* Code Comparison */}
      <div className="mt-6">
        <button
          onClick={() => setShowCode(!showCode)}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded transition-colors"
        >
          {showCode ? 'Hide' : 'Show'} GPU Code Comparison
        </button>

        <AnimatePresence>
          {showCode && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4"
            >
              <div>
                <h3 className="text-lg font-semibold mb-2">WebGPU Compute Shader</h3>
                <div className="h-96 overflow-hidden rounded-lg">
                  <MonacoEditor
                    value={shaderCode}
                    language="rust"
                    theme="vs-dark"
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      fontSize: 12
                    }}
                  />
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">CUDA Kernel (Reference)</h3>
                <div className="h-96 overflow-hidden rounded-lg">
                  <MonacoEditor
                    value={cudaCode}
                    language="cpp"
                    theme="vs-dark"
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      fontSize: 12
                    }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Volatility Surface */}
      <div className="mt-6">
        <button
          onClick={() => setShowVolatilitySurface(!showVolatilitySurface)}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded transition-colors"
        >
          {showVolatilitySurface ? 'Hide' : 'Show'} Volatility Surface
        </button>

        <AnimatePresence>
          {showVolatilitySurface && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700"
            >
              <h3 className="text-xl font-semibold mb-4">Implied Volatility Surface</h3>
              <Plot
                data={[
                  {
                    type: 'surface',
                    z: generateVolatilitySurface().surface,
                    x: generateVolatilitySurface().strikes,
                    y: generateVolatilitySurface().maturities,
                    colorscale: 'Viridis'
                  }
                ]}
                layout={{
                  height: 500,
                  scene: {
                    xaxis: { title: { text: 'Strike Price' } },
                    yaxis: { title: { text: 'Time to Maturity (years)' } },
                    zaxis: { title: { text: 'Implied Volatility' } },
                    camera: {
                      eye: { x: 1.5, y: 1.5, z: 1.5 }
                    }
                  },
                  paper_bgcolor: 'transparent',
                  font: { color: '#9ca3af' }
                }}
                config={{ displayModeBar: false }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}