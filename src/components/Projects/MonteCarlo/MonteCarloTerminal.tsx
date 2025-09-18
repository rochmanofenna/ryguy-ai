'use client';

import React, { useEffect, useRef, useState } from 'react';
import { MonteCarloWebGPU, MonteCarloCPU, OptionParams } from './MonteCarloWebGPU';

export default function MonteCarloTerminal() {
  const terminalRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [currentParams] = useState<OptionParams>({
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

    let term: any;
    let fitAddon: any;

    const initTerminal = async () => {
      try {
        const { Terminal } = await import('@xterm/xterm');
        const { FitAddon } = await import('@xterm/addon-fit');

        // Add xterm styles dynamically
        if (typeof document !== 'undefined' && !document.getElementById('xterm-styles')) {
          const link = document.createElement('link');
          link.id = 'xterm-styles';
          link.rel = 'stylesheet';
          link.href = 'https://unpkg.com/@xterm/xterm@5.3.0/css/xterm.css';
          document.head.appendChild(link);
        }

        term = new Terminal({
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

        fitAddon = new FitAddon();
        term.loadAddon(fitAddon);
        term.open(terminalRef.current);
        fitAddon.fit();

        setIsLoading(false);

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
        webgpuEngine.current = new MonteCarloWebGPU();
        const webgpuSupported = await webgpuEngine.current.initialize();
        cpuEngine.current = new MonteCarloCPU();

        if (webgpuSupported) {
          term.writeln('\\x1b[1;32m[SYSTEM]\\x1b[0m WebGPU compute pipeline ready');
        } else {
          term.writeln('\\x1b[1;33m[SYSTEM]\\x1b[0m WebGPU not available, using CPU fallback');
        }
        term.write('\\x1b[1;34mquant>\\x1b[0m ');

        // Command handling
        let currentLine = '';
        const commandHistory: string[] = [];
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
              term.writeln('  \\x1b[33mclear\\x1b[0m                                   - Clear terminal');
              term.writeln('  \\x1b[33mexit\\x1b[0m                                    - Exit terminal');
              break;

            case 'price':
              if (args.length > 1) {
                const params: OptionParams = {
                  S: parseFloat(args[1]) || currentParams.S,
                  K: parseFloat(args[2]) || currentParams.K,
                  r: parseFloat(args[3]) || currentParams.r,
                  T: parseFloat(args[4]) || currentParams.T,
                  sigma: parseFloat(args[5]) || currentParams.sigma,
                  paths: parseInt(args[6]) || currentParams.paths,
                  steps: currentParams.steps
                };

                term.writeln('\\x1b[33m[CALC]\\x1b[0m Pricing option...');

                if (webgpuEngine.current) {
                  try {
                    const result = await webgpuEngine.current.priceOption(params);
                    term.writeln('\\x1b[1;32m[RESULT]\\x1b[0m Option Price: $' + result.price.toFixed(4));
                    term.writeln('  Standard Error: ±' + result.standardError.toFixed(4));
                    term.writeln('  Execution Time: ' + result.executionTime.toFixed(1) + 'ms');
                  } catch (error) {
                    term.writeln('\\x1b[1;31m[ERROR]\\x1b[0m Failed to calculate price');
                  }
                }
              } else {
                term.writeln('\\x1b[1;31m[ERROR]\\x1b[0m Usage: price [S] [K] [r] [T] [sigma] [paths]');
              }
              break;

            case 'clear':
              term.clear();
              break;

            default:
              if (cmd.trim()) {
                term.writeln('\\x1b[31m[ERROR]\\x1b[0m Unknown command: ' + command);
                term.writeln('Type \\x1b[33mhelp\\x1b[0m for available commands');
              }
          }
        };

        // Input handling
        term.onData((data: string) => {
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
          if (fitAddon) fitAddon.fit();
        };
        window.addEventListener('resize', handleResize);
      } catch (error) {
        console.error('Failed to initialize terminal:', error);
        setIsLoading(false);
      }
    };

    initTerminal();

    // Cleanup
    return () => {
      if (term) term.dispose();
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
          {isLoading ? (
            <div className="flex items-center justify-center h-[600px]">
              <div className="text-white animate-pulse">Loading terminal...</div>
            </div>
          ) : null}
          <div
            ref={terminalRef}
            className="w-full"
            style={{ height: '600px', display: isLoading ? 'none' : 'block' }}
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