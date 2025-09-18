'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';
import { AdvancedMonteCarlo, AdvancedOptionParams, ConvergenceData } from './AdvancedMonteCarlo';

const Plot = dynamic(() => import('react-plotly.js').then(mod => mod.default), {
  ssr: false,
  loading: () => <div className="h-64 bg-slate-800/50 animate-pulse rounded" />
});

export default function RealisticMonteCarloDemo() {
  const [params, setParams] = useState<AdvancedOptionParams>({
    S: 100,
    K: 100,
    r: 0.05,
    T: 0.25,
    sigma: 0.25,
    q: 0.02,
    paths: 10000,
    steps: 63, // Quarterly option with daily steps
    optionType: 'call',
    exerciseType: 'european',
    varianceReduction: 'both'
  });

  const [results, setResults] = useState<any>(null);
  const [greeks, setGreeks] = useState<any>(null);
  const [convergenceData, setConvergenceData] = useState<ConvergenceData | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [activeTab, setActiveTab] = useState<'pricing' | 'greeks' | 'convergence' | 'volatility'>('pricing');

  const engine = new AdvancedMonteCarlo();

  const calculatePrice = useCallback(async () => {
    setIsCalculating(true);
    try {
      const startTime = performance.now();

      if (params.exerciseType === 'american') {
        const result = await engine.priceAmerican(params);
        setResults({
          ...result,
          executionTime: performance.now() - startTime,
          device: 'cpu'
        });
      } else {
        const result = await engine.priceEuropean(params);
        setResults({
          ...result,
          executionTime: performance.now() - startTime,
          device: 'cpu'
        });
      }

      // Calculate Greeks in background
      if (params.exerciseType === 'european') {
        const greekValues = await engine.calculateGreeks(params);
        setGreeks(greekValues);
      }
    } catch (error) {
      console.error('Pricing failed:', error);
    } finally {
      setIsCalculating(false);
    }
  }, [params]);

  const runConvergenceAnalysis = async () => {
    setIsCalculating(true);
    try {
      const data = await engine.analyzeConvergence(params);
      setConvergenceData(data);
      setActiveTab('convergence');
    } catch (error) {
      console.error('Convergence analysis failed:', error);
    } finally {
      setIsCalculating(false);
    }
  };

  // Auto-calculate on parameter change
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isCalculating) {
        calculatePrice();
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [params]);

  // Generate implied volatility smile
  const generateVolatilitySmile = () => {
    const strikes = Array.from({ length: 21 }, (_, i) => 80 + i * 2);
    const impliedVols = strikes.map(K => {
      const moneyness = K / params.S;
      // Realistic vol smile shape
      const atmVol = params.sigma;
      const skew = -0.1;
      const convexity = 0.05;

      const logMoneyness = Math.log(moneyness);
      const iv = atmVol + skew * logMoneyness + convexity * Math.pow(logMoneyness, 2);

      return Math.max(0.05, Math.min(0.8, iv));
    });

    return { strikes, impliedVols };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Advanced Monte Carlo Option Pricing
        </h1>
        <p className="text-gray-400">
          Production-grade implementation with variance reduction, American exercise, and Greeks
        </p>
        <div className="flex gap-4 mt-4 text-sm">
          <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded">
            Variance Reduction: {params.varianceReduction}
          </span>
          <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded">
            Exercise: {params.exerciseType}
          </span>
          {results?.varianceReduction && (
            <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded">
              Variance ↓ {results.varianceReduction.toFixed(1)}%
            </span>
          )}
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {(['pricing', 'greeks', 'convergence', 'volatility'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded transition-all capitalize ${
              activeTab === tab
                ? 'bg-cyan-500 text-white'
                : 'bg-slate-700 text-gray-400 hover:bg-slate-600'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Panel */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-semibold mb-4">Parameters</h2>

          <div className="space-y-4">
            {/* Option Type */}
            <div>
              <label className="text-sm text-gray-400 mb-2 block">Option Type</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setParams({ ...params, optionType: 'call' })}
                  className={`px-3 py-1 rounded text-sm ${
                    params.optionType === 'call'
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-700 text-gray-400'
                  }`}
                >
                  Call
                </button>
                <button
                  onClick={() => setParams({ ...params, optionType: 'put' })}
                  className={`px-3 py-1 rounded text-sm ${
                    params.optionType === 'put'
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-700 text-gray-400'
                  }`}
                >
                  Put
                </button>
              </div>
            </div>

            {/* Exercise Type */}
            <div>
              <label className="text-sm text-gray-400 mb-2 block">Exercise Type</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setParams({ ...params, exerciseType: 'european' })}
                  className={`px-3 py-1 rounded text-sm ${
                    params.exerciseType === 'european'
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-700 text-gray-400'
                  }`}
                >
                  European
                </button>
                <button
                  onClick={() => setParams({ ...params, exerciseType: 'american' })}
                  className={`px-3 py-1 rounded text-sm ${
                    params.exerciseType === 'american'
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-700 text-gray-400'
                  }`}
                >
                  American
                </button>
              </div>
            </div>

            {/* Variance Reduction */}
            <div>
              <label className="text-sm text-gray-400 mb-2 block">Variance Reduction</label>
              <select
                value={params.varianceReduction}
                onChange={(e) => setParams({ ...params, varianceReduction: e.target.value as any })}
                className="w-full bg-slate-700 text-white rounded px-3 py-1"
              >
                <option value="none">None</option>
                <option value="antithetic">Antithetic Variates</option>
                <option value="control">Control Variates</option>
                <option value="both">Both Techniques</option>
              </select>
            </div>

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
                max="0.8"
                step="0.01"
                value={params.sigma}
                onChange={(e) => setParams({ ...params, sigma: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Time to Maturity */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Time to Maturity</span>
                <span className="text-cyan-400">{(params.T * 12).toFixed(1)} months</span>
              </label>
              <input
                type="range"
                min="0.08"
                max="2"
                step="0.08"
                value={params.T}
                onChange={(e) => setParams({ ...params, T: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>

            {/* Monte Carlo Paths */}
            <div>
              <label className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Simulation Paths</span>
                <span className="text-cyan-400">{params.paths.toLocaleString()}</span>
              </label>
              <input
                type="range"
                min="1000"
                max="100000"
                step="1000"
                value={params.paths}
                onChange={(e) => setParams({ ...params, paths: Number(e.target.value) })}
                className="w-full accent-cyan-400"
              />
            </div>
          </div>

          <button
            onClick={runConvergenceAnalysis}
            disabled={isCalculating}
            className="w-full mt-4 px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded transition-all hover:opacity-90 disabled:opacity-50"
          >
            {isCalculating ? 'Calculating...' : 'Run Convergence Analysis'}
          </button>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {activeTab === 'pricing' && (
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
                      SE: ±{results.standardError.toFixed(4)}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-400">95% Confidence</p>
                    <p className="text-lg text-gray-300">
                      [{results.confidence95?.[0].toFixed(3)}, {results.confidence95?.[1].toFixed(3)}]
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
                      <span className="text-xs ml-1">
                        ({params.S > params.K ? 'ITM' : params.S < params.K ? 'OTM' : 'ATM'})
                      </span>
                    </p>
                  </div>

                  {results.earlyExercisePremium !== undefined && (
                    <div>
                      <p className="text-sm text-gray-400">Early Exercise Premium</p>
                      <p className="text-lg text-purple-400">
                        ${results.earlyExercisePremium.toFixed(4)}
                      </p>
                    </div>
                  )}

                  {results.varianceReduction !== undefined && (
                    <div>
                      <p className="text-sm text-gray-400">Variance Reduction</p>
                      <p className="text-lg text-purple-400">
                        {results.varianceReduction.toFixed(1)}%
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-32">
                  <div className="text-gray-500">
                    {isCalculating ? 'Calculating...' : 'Waiting for calculation'}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'greeks' && greeks && (
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4">Option Greeks</h2>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Δ Delta</p>
                  <p className="text-xl font-semibold text-blue-400">{greeks.delta.toFixed(4)}</p>
                  <p className="text-xs text-gray-500">Price/Spot</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Γ Gamma</p>
                  <p className="text-xl font-semibold text-green-400">{greeks.gamma.toFixed(6)}</p>
                  <p className="text-xs text-gray-500">Delta/Spot</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">ν Vega</p>
                  <p className="text-xl font-semibold text-purple-400">{greeks.vega.toFixed(4)}</p>
                  <p className="text-xs text-gray-500">Price/Vol%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Θ Theta</p>
                  <p className="text-xl font-semibold text-red-400">{greeks.theta.toFixed(4)}</p>
                  <p className="text-xs text-gray-500">Price/Day</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">ρ Rho</p>
                  <p className="text-xl font-semibold text-yellow-400">{greeks.rho.toFixed(4)}</p>
                  <p className="text-xs text-gray-500">Price/Rate%</p>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-3">Delta Profile</h3>
                <Plot
                  data={[{
                    x: Array.from({ length: 50 }, (_, i) => 60 + i * 2),
                    y: Array.from({ length: 50 }, (_, i) => {
                      const S = 60 + i * 2;
                      const moneyness = S / params.K;
                      // Approximate delta profile
                      if (params.optionType === 'call') {
                        return 1 / (1 + Math.exp(-5 * (moneyness - 1)));
                      } else {
                        return -1 / (1 + Math.exp(5 * (moneyness - 1)));
                      }
                    }),
                    type: 'scatter',
                    mode: 'lines',
                    marker: { color: '#06b6d4' },
                    name: 'Delta'
                  }]}
                  layout={{
                    height: 250,
                    margin: { t: 10, b: 40, l: 50, r: 10 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#9ca3af' },
                    xaxis: {
                      gridcolor: '#374151',
                      title: { text: 'Spot Price' }
                    },
                    yaxis: {
                      gridcolor: '#374151',
                      title: { text: 'Delta' }
                    }
                  }}
                  config={{ displayModeBar: false }}
                />
              </div>
            </div>
          )}

          {activeTab === 'convergence' && convergenceData && (
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4">Convergence Analysis</h2>

              <Plot
                data={[
                  {
                    x: convergenceData.pathCounts,
                    y: convergenceData.prices,
                    error_y: {
                      type: 'data',
                      array: convergenceData.standardErrors.map(se => se * 1.96),
                      visible: true,
                      color: '#94a3b8'
                    },
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: '#06b6d4', size: 8 },
                    line: { width: 2 },
                    name: 'Option Price'
                  }
                ]}
                layout={{
                  height: 350,
                  margin: { t: 20, b: 50, l: 60, r: 20 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#9ca3af' },
                  xaxis: {
                    gridcolor: '#374151',
                    title: { text: 'Number of Paths' },
                    type: 'log'
                  },
                  yaxis: {
                    gridcolor: '#374151',
                    title: { text: 'Option Price ($)' }
                  }
                }}
                config={{ displayModeBar: false }}
              />

              <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                <div className="bg-slate-700/50 rounded p-3">
                  <p className="text-gray-400">Convergence Rate</p>
                  <p className="text-cyan-400">O(1/√n)</p>
                </div>
                <div className="bg-slate-700/50 rounded p-3">
                  <p className="text-gray-400">Final Std Error</p>
                  <p className="text-green-400">
                    ±{convergenceData.standardErrors[convergenceData.standardErrors.length - 1].toFixed(5)}
                  </p>
                </div>
                <div className="bg-slate-700/50 rounded p-3">
                  <p className="text-gray-400">Paths for 0.01 SE</p>
                  <p className="text-purple-400">
                    ~{Math.round(Math.pow(convergenceData.standardErrors[0] / 0.01, 2) * 1000).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'volatility' && (
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4">Implied Volatility Smile</h2>

              <Plot
                data={[{
                  x: generateVolatilitySmile().strikes,
                  y: generateVolatilitySmile().impliedVols.map(v => v * 100),
                  type: 'scatter',
                  mode: 'lines+markers',
                  marker: { color: '#a855f7', size: 6 },
                  line: { width: 2, shape: 'spline' },
                  name: 'IV Smile'
                }]}
                layout={{
                  height: 350,
                  margin: { t: 20, b: 50, l: 60, r: 20 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#9ca3af' },
                  xaxis: {
                    gridcolor: '#374151',
                    title: { text: 'Strike Price ($)' }
                  },
                  yaxis: {
                    gridcolor: '#374151',
                    title: { text: 'Implied Volatility (%)' }
                  },
                  shapes: [{
                    type: 'line',
                    x0: params.K,
                    x1: params.K,
                    y0: 0,
                    y1: 100,
                    line: { color: '#ef4444', width: 1, dash: 'dot' }
                  }]
                }}
                config={{ displayModeBar: false }}
              />

              <div className="mt-4 text-sm text-gray-400">
                <p>The volatility smile shows higher implied volatilities for OTM options,</p>
                <p>reflecting market pricing of tail risk and volatility skew.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}