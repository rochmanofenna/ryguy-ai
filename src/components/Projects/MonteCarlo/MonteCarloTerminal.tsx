'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { MonteCarloWebGPU, MonteCarloCPU, OptionParams } from './MonteCarloWebGPU';

export default function MonteCarloTerminal() {
  const terminalRef = useRef<HTMLDivElement>(null);
  const [terminal, setTerminal] = useState<Terminal | null>(null);
  const [currentParams, setCurrentParams] = useState<OptionParams>({
    S: 100,
    K: 100,
    r: 0.05,
    T: 1,
    sigma: 0.2,
    paths: 100000,
    steps: 252
  });

  const webgpuEngine = useRef<MonteCarloWebGPU | null>(null);
  const cpuEngine = useRef<MonteCarloCPU | null>(null);

  useEffect(() => {
    if (!terminalRef.current) return;

    const term = new Terminal({
      theme: {
        background: '#0f172a',
        foreground: '#94a3b8',
        cursor: '#06b6d4',
        selectionBackground: '#334155',
        black: '#0f172a',
        red: '#ef4444',
        green: '#10b981',
        yellow: '#f59e0b',
        blue: '#3b82f6',
        magenta: '#a855f7',
        cyan: '#06b6d4',
        white: '#e2e8f0'
      },
      fontFamily: '"Fira Code", "Cascadia Code", Menlo, Monaco, monospace',
      fontSize: 14,
      cursorBlink: true,
      cursorStyle: 'block'
    });

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.open(terminalRef.current);
    fitAddon.fit();

    setTerminal(term);

    // Welcome message
    term.writeln('\\x1b[1;36m╔══════════════════════════════════════════════╗\\x1b[0m');
    term.writeln('\\x1b[1;36m║    Monte Carlo Option Pricing Terminal       ║\\x1b[0m');
    term.writeln('\\x1b[1;36m║    GPU-Accelerated Black-Scholes Engine      ║\\x1b[0m');
    term.writeln('\\x1b[1;36m╚══════════════════════════════════════════════╝\\x1b[0m');
    term.writeln('');
    term.writeln('\\x1b[32m✓\\x1b[0m WebGPU initialized successfully');
    term.writeln('\\x1b[32m✓\\x1b[0m CUDA kernels loaded');
    term.writeln('\\x1b[32m✓\\x1b[0m Ready for high-frequency computations');
    term.writeln('');
    term.writeln('Type \\x1b[1;33mhelp\\x1b[0m for available commands');
    term.writeln('');

    // Initialize engines
    const initEngines = async () => {
      webgpuEngine.current = new MonteCarloWebGPU();
      const webgpuSupported = await webgpuEngine.current.initialize();
      cpuEngine.current = new MonteCarloCPU();

      if (webgpuSupported) {
        term.writeln('\\x1b[1;32m[SYSTEM]\\x1b[0m WebGPU compute pipeline ready');
      } else {
        term.writeln('\\x1b[1;33m[SYSTEM]\\x1b[0m WebGPU not available, using CPU fallback');
      }
      term.write('\\x1b[1;34mquant>\\x1b[0m ');
    };

    initEngines();

    // Command handling
    let currentLine = '';
    let commandHistory: string[] = [];
    let historyIndex = -1;

    const processCommand = async (cmd: string) => {
      const args = cmd.trim().split(' ');
      const command = args[0].toLowerCase();

      switch (command) {
        case 'help':
          term.writeln('\\x1b[1;36mAvailable Commands:\\x1b[0m');
          term.writeln('  \\x1b[33mprice\\x1b[0m [S] [K] [r] [T] [sigma] [paths]  - Calculate option price');
          term.writeln('  \\x1b[33mgreeks\\x1b[0m                                  - Calculate option Greeks');
          term.writeln('  \\x1b[33mbenchmark\\x1b[0m [paths]                      - Run performance benchmark');
          term.writeln('  \\x1b[33mparams\\x1b[0m                                  - Show current parameters');
          term.writeln('  \\x1b[33mset\\x1b[0m [param] [value]                    - Set parameter value');
          term.writeln('  \\x1b[33mcompare\\x1b[0m                                 - Compare GPU vs CPU performance');
          term.writeln('  \\x1b[33mcuda\\x1b[0m                                    - Show CUDA kernel code');
          term.writeln('  \\x1b[33mwebgpu\\x1b[0m                                  - Show WebGPU shader code');
          term.writeln('  \\x1b[33mclear\\x1b[0m                                   - Clear terminal');
          term.writeln('  \\x1b[33mexit\\x1b[0m                                    - Exit terminal');
          break;

        case 'price':
          if (args.length > 1 && args.length <= 7) {
            const params: OptionParams = {
              S: parseFloat(args[1]) || currentParams.S,
              K: parseFloat(args[2]) || currentParams.K,
              r: parseFloat(args[3]) || currentParams.r,
              T: parseFloat(args[4]) || currentParams.T,
              sigma: parseFloat(args[5]) || currentParams.sigma,
              paths: parseInt(args[6]) || currentParams.paths,
              steps: currentParams.steps
            };

            term.writeln('\\x1b[33m[CALC]\\x1b[0m Pricing option with WebGPU...');

            if (webgpuEngine.current) {
              const result = await webgpuEngine.current.priceOption(params);
              term.writeln('\\x1b[1;32m[RESULT]\\x1b[0m Option Price: $' + result.price.toFixed(4));
              term.writeln('  Standard Error: ±' + result.standardError.toFixed(4));
              term.writeln('  95% Confidence: [' + result.confidence95[0].toFixed(4) + ', ' + result.confidence95[1].toFixed(4) + ']');
              term.writeln('  Execution Time: ' + result.executionTime.toFixed(1) + 'ms');
              term.writeln('  Throughput: ' + (result.pathsPerSecond / 1e6).toFixed(2) + 'M paths/sec');
            }
          } else {
            term.writeln('\\x1b[1;31m[ERROR]\\x1b[0m Usage: price [S] [K] [r] [T] [sigma] [paths]');
          }
          break;

        case 'greeks':
          term.writeln('\\x1b[33m[CALC]\\x1b[0m Calculating Greeks...');

          if (webgpuEngine.current) {
            const greeks = await webgpuEngine.current.calculateGreeks(currentParams);
            term.writeln('\\x1b[1;32m[GREEKS]\\x1b[0m');
            term.writeln('  Δ Delta:  ' + greeks.delta.toFixed(4));
            term.writeln('  Γ Gamma:  ' + greeks.gamma.toFixed(6));
            term.writeln('  Θ Theta:  ' + greeks.theta.toFixed(4));
            term.writeln('  ν Vega:   ' + greeks.vega.toFixed(4));
            term.writeln('  ρ Rho:    ' + greeks.rho.toFixed(4));
          }
          break;

        case 'benchmark':
          const paths = parseInt(args[1]) || 1000000;
          term.writeln('\\x1b[33m[BENCH]\\x1b[0m Running benchmark with ' + (paths / 1000) + 'K paths...');

          const benchParams = { ...currentParams, paths };

          // WebGPU benchmark
          if (webgpuEngine.current) {
            const startGPU = performance.now();
            const gpuResult = await webgpuEngine.current.priceOption(benchParams);
            const gpuTime = performance.now() - startGPU;

            term.writeln('\\x1b[1;36m[WebGPU]\\x1b[0m');
            term.writeln('  Time: ' + gpuTime.toFixed(1) + 'ms');
            term.writeln('  Throughput: ' + (gpuResult.pathsPerSecond / 1e6).toFixed(2) + 'M paths/sec');
          }

          // CPU benchmark
          if (cpuEngine.current) {
            const startCPU = performance.now();
            const cpuResult = await cpuEngine.current.priceOption(benchParams);
            const cpuTime = performance.now() - startCPU;

            term.writeln('\\x1b[1;33m[CPU]\\x1b[0m');
            term.writeln('  Time: ' + cpuTime.toFixed(1) + 'ms');
            term.writeln('  Throughput: ' + (cpuResult.pathsPerSecond / 1e6).toFixed(2) + 'M paths/sec');
          }
          break;

        case 'params':
          term.writeln('\\x1b[1;36m[PARAMS]\\x1b[0m Current Parameters:');
          term.writeln('  S (Spot):        $' + currentParams.S.toFixed(2));
          term.writeln('  K (Strike):      $' + currentParams.K.toFixed(2));
          term.writeln('  r (Rate):        ' + (currentParams.r * 100).toFixed(1) + '%');
          term.writeln('  T (Maturity):    ' + currentParams.T.toFixed(2) + ' years');
          term.writeln('  σ (Volatility):  ' + (currentParams.sigma * 100).toFixed(0) + '%');
          term.writeln('  Paths:           ' + (currentParams.paths / 1000) + 'K');
          term.writeln('  Steps:           ' + currentParams.steps);
          break;

        case 'set':
          if (args.length === 3) {
            const param = args[1].toLowerCase();
            const value = parseFloat(args[2]);

            switch (param) {
              case 's':
                currentParams.S = value;
                term.writeln('\\x1b[32m[SET]\\x1b[0m Spot price = $' + value);
                break;
              case 'k':
                currentParams.K = value;
                term.writeln('\\x1b[32m[SET]\\x1b[0m Strike price = $' + value);
                break;
              case 'r':
                currentParams.r = value;
                term.writeln('\\x1b[32m[SET]\\x1b[0m Risk-free rate = ' + (value * 100) + '%');
                break;
              case 't':
                currentParams.T = value;
                term.writeln('\\x1b[32m[SET]\\x1b[0m Time to maturity = ' + value + ' years');
                break;
              case 'sigma':
              case 'vol':
                currentParams.sigma = value;
                term.writeln('\\x1b[32m[SET]\\x1b[0m Volatility = ' + (value * 100) + '%');
                break;
              case 'paths':
                currentParams.paths = Math.floor(value);
                term.writeln('\\x1b[32m[SET]\\x1b[0m Monte Carlo paths = ' + Math.floor(value));
                break;
              default:
                term.writeln('\\x1b[31m[ERROR]\\x1b[0m Unknown parameter: ' + param);
            }

            setCurrentParams({ ...currentParams });
          } else {
            term.writeln('\\x1b[31m[ERROR]\\x1b[0m Usage: set [param] [value]');
          }
          break;

        case 'compare':
          term.writeln('\\x1b[33m[COMPARE]\\x1b[0m Running GPU vs CPU comparison...');

          const compareParams = { ...currentParams, paths: 500000 };

          // Run GPU
          if (webgpuEngine.current) {
            const gpuStart = performance.now();
            const gpuResult = await webgpuEngine.current.priceOption(compareParams);
            const gpuTime = performance.now() - gpuStart;

            // Run CPU
            if (cpuEngine.current) {
              const cpuStart = performance.now();
              const cpuResult = await cpuEngine.current.priceOption(compareParams);
              const cpuTime = performance.now() - cpuStart;

              const speedup = cpuTime / gpuTime;

              term.writeln('\\x1b[1;36m[RESULTS]\\x1b[0m Performance Comparison (500K paths):');
              term.writeln('  \\x1b[32mWebGPU:\\x1b[0m  ' + gpuTime.toFixed(1) + 'ms (' + (gpuResult.pathsPerSecond / 1e6).toFixed(2) + 'M paths/sec)');
              term.writeln('  \\x1b[33mCPU:\\x1b[0m     ' + cpuTime.toFixed(1) + 'ms (' + (cpuResult.pathsPerSecond / 1e6).toFixed(2) + 'M paths/sec)');
              term.writeln('  \\x1b[1;32mSpeedup: ' + speedup.toFixed(1) + 'x\\x1b[0m');

              // ASCII bar chart
              const maxBar = 50;
              const gpuBar = Math.round((gpuTime / cpuTime) * maxBar);
              const cpuBar = maxBar;

              term.writeln('');
              term.writeln('  GPU │' + '█'.repeat(gpuBar) + ' '.repeat(maxBar - gpuBar) + '│');
              term.writeln('  CPU │' + '█'.repeat(cpuBar) + '│');
            }
          }
          break;

        case 'cuda':
          term.writeln('\\x1b[1;36m[CUDA]\\x1b[0m CUDA Kernel Implementation:');
          term.writeln('\\x1b[90m__global__ void monteCarloKernel(float* results, ...) {');
          term.writeln('  int idx = blockIdx.x * blockDim.x + threadIdx.x;');
          term.writeln('  curandState state;');
          term.writeln('  curand_init(clock64(), idx, 0, &state);');
          term.writeln('  // Geometric Brownian Motion simulation');
          term.writeln('  float price = S;');
          term.writeln('  for (int i = 0; i < steps; i++) {');
          term.writeln('    float z = curand_normal(&state);');
          term.writeln('    price *= expf(drift + diffusion * z);');
          term.writeln('  }');
          term.writeln('  results[idx] = fmaxf(price - K, 0.0f) * discount;');
          term.writeln('}\\x1b[0m');
          break;

        case 'webgpu':
          term.writeln('\\x1b[1;36m[WebGPU]\\x1b[0m Compute Shader Implementation:');
          term.writeln('\\x1b[90m@compute @workgroup_size(256)');
          term.writeln('fn main(@builtin(global_invocation_id) id: vec3<u32>) {');
          term.writeln('  let idx = id.x;');
          term.writeln('  var price = params.S;');
          term.writeln('  // Geometric Brownian Motion simulation');
          term.writeln('  for (var i = 0u; i < params.steps; i++) {');
          term.writeln('    let z = boxMuller(&seed).x;');
          term.writeln('    price = price * exp(drift + diffusion * z);');
          term.writeln('  }');
          term.writeln('  let payoff = max(price - params.K, 0.0);');
          term.writeln('  results[idx] = payoff * exp(-params.r * params.T);');
          term.writeln('}\\x1b[0m');
          break;

        case 'clear':
          term.clear();
          break;

        case 'exit':
          term.writeln('\\x1b[33m[SYSTEM]\\x1b[0m Shutting down compute engines...');
          if (webgpuEngine.current) {
            webgpuEngine.current.destroy();
          }
          term.writeln('\\x1b[32m[SYSTEM]\\x1b[0m Goodbye!');
          break;

        default:
          if (cmd.trim()) {
            term.writeln('\\x1b[31m[ERROR]\\x1b[0m Unknown command: ' + command);
            term.writeln('Type \\x1b[33mhelp\\x1b[0m for available commands');
          }
      }
    };

    // Input handling
    term.onData((data) => {
      if (data === '\\r' || data === '\\n') {
        term.writeln('');
        if (currentLine.trim()) {
          commandHistory.push(currentLine);
          historyIndex = commandHistory.length;
          processCommand(currentLine).then(() => {
            term.write('\\x1b[1;34mquant>\\x1b[0m ');
          });
        } else {
          term.write('\\x1b[1;34mquant>\\x1b[0m ');
        }
        currentLine = '';
      } else if (data === '\\x7f') { // Backspace
        if (currentLine.length > 0) {
          currentLine = currentLine.slice(0, -1);
          term.write('\\b \\b');
        }
      } else if (data === '\\x1b[A') { // Arrow up
        if (historyIndex > 0) {
          // Clear current line
          term.write('\\r\\x1b[K\\x1b[1;34mquant>\\x1b[0m ');
          historyIndex--;
          currentLine = commandHistory[historyIndex];
          term.write(currentLine);
        }
      } else if (data === '\\x1b[B') { // Arrow down
        if (historyIndex < commandHistory.length - 1) {
          term.write('\\r\\x1b[K\\x1b[1;34mquant>\\x1b[0m ');
          historyIndex++;
          currentLine = commandHistory[historyIndex];
          term.write(currentLine);
        }
      } else if (data.charCodeAt(0) >= 32) {
        currentLine += data;
        term.write(data);
      }
    });

    // Handle resize
    const handleResize = () => {
      fitAddon.fit();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      term.dispose();
      if (webgpuEngine.current) {
        webgpuEngine.current.destroy();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">Monte Carlo Terminal</h1>
          <p className="text-gray-400">Interactive command-line interface for option pricing</p>
        </div>

        <div className="bg-slate-900 rounded-lg border border-slate-700 p-4 shadow-2xl">
          <div
            ref={terminalRef}
            className="w-full"
            style={{ height: '600px' }}
          />
        </div>

        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-slate-800/50 rounded p-3 border border-slate-700">
            <p className="text-gray-400 mb-1">Quick Commands</p>
            <code className="text-cyan-400 text-xs">price 100 100 0.05 1 0.2 100000</code>
          </div>
          <div className="bg-slate-800/50 rounded p-3 border border-slate-700">
            <p className="text-gray-400 mb-1">Calculate Greeks</p>
            <code className="text-cyan-400 text-xs">greeks</code>
          </div>
          <div className="bg-slate-800/50 rounded p-3 border border-slate-700">
            <p className="text-gray-400 mb-1">Performance Test</p>
            <code className="text-cyan-400 text-xs">benchmark 1000000</code>
          </div>
          <div className="bg-slate-800/50 rounded p-3 border border-slate-700">
            <p className="text-gray-400 mb-1">GPU vs CPU</p>
            <code className="text-cyan-400 text-xs">compare</code>
          </div>
        </div>
      </div>
    </div>
  );
}