'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TerminalLine, Command } from '@/types';
import CommandPalette from './CommandPalette';
import befService from '@/services/befService';

const INITIAL_PROMPT = 'ryan@rochmanofenna:~$';

// Import original commands
const commands: Record<string, Command> = {
  help: {
    name: 'help',
    description: 'Show available commands',
    handler: () => {}
  },
  portfolio: {
    name: 'portfolio',
    description: 'Live trading P&L dashboard (BEF Pipeline)',
    handler: () => {}
  },
  gpu: {
    name: 'gpu',
    description: 'BICEP Monte Carlo demonstration',
    handler: () => {}
  },
  pipeline: {
    name: 'pipeline',
    description: 'Run complete BEF pipeline',
    handler: () => {}
  },
  enn: {
    name: 'enn',
    description: 'ENN committor prediction',
    handler: () => {}
  },
  fusion: {
    name: 'fusion',
    description: 'FusionAlpha planning system',
    handler: () => {}
  },
  stream: {
    name: 'stream',
    description: 'Real-time BEF pipeline stream',
    handler: () => {}
  },
  clear: {
    name: 'clear',
    description: 'Clear terminal',
    handler: () => {}
  }
};

export default function TerminalEnhanced() {
  const [lines, setLines] = useState<TerminalLine[]>([
    {
      id: 'welcome',
      content: `${INITIAL_PROMPT} whoami`,
      type: 'input',
      timestamp: new Date()
    },
    {
      id: 'whoami-output',
      content: '> BEF Pipeline Integration | BICEP + ENN + FusionAlpha',
      type: 'output',
      timestamp: new Date()
    },
    {
      id: 'status',
      content: '> Real Mathematical Models | Sub-ms Latency | Research Grade',
      type: 'output',
      timestamp: new Date()
    },
    {
      id: 'hint',
      content: "> Type 'help' or press Cmd+K to navigate",
      type: 'system',
      timestamp: new Date()
    }
  ]);

  const [currentInput, setCurrentInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [streamConnection, setStreamConnection] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  const addLine = useCallback((content: string, type: TerminalLine['type'] = 'output') => {
    const newLine: TerminalLine = {
      id: `${Date.now()}-${Math.random()}`,
      content,
      type,
      timestamp: new Date()
    };
    setLines(prev => [...prev, newLine]);
  }, []);

  const executeCommand = useCallback(async (input: string) => {
    const trimmedInput = input.trim().toLowerCase();
    const [commandName, ...args] = trimmedInput.split(' ');

    // Add input line
    addLine(`${INITIAL_PROMPT} ${input}`, 'input');

    if (!trimmedInput) {
      return;
    }

    switch (commandName) {
      case 'help':
        addLine('Available BEF Pipeline Commands:', 'output');
        Object.values(commands).forEach(cmd => {
          addLine(`  ${cmd.name.padEnd(12)} - ${cmd.description}`, 'output');
        });
        addLine('', 'output');
        addLine('Mathematical Components:', 'system');
        addLine('  BICEP     - SDE trajectory generation (Rust)', 'system');
        addLine('  ENN       - Entangled neural networks (C++)', 'system');
        addLine('  FusionAlpha - Committor-based planning', 'system');
        break;

      case 'portfolio':
        setIsTyping(true);
        addLine('🔄 Fetching real-time metrics from BEF pipeline...', 'system');

        try {
          const metrics = await befService.getPortfolioMetrics();
          addLine('=== LIVE PORTFOLIO METRICS (BEF Pipeline) ===', 'output');
          addLine('', 'output');
          addLine('📊 REAL-TIME PERFORMANCE:', 'output');
          addLine(`• Current P&L: ${metrics.current_pnl >= 0 ? '+' : ''}$${Math.abs(metrics.current_pnl).toFixed(0)}`, 'output');
          addLine(`• Daily P&L: ${metrics.daily_pnl >= 0 ? '+' : ''}$${Math.abs(metrics.daily_pnl).toFixed(0)}`, 'output');
          addLine(`• Sharpe Ratio: ${metrics.sharpe_ratio.toFixed(2)}`, 'output');
          addLine(`• Max Drawdown: ${metrics.max_drawdown.toFixed(1)}%`, 'output');
          addLine(`• Win Rate: ${metrics.win_rate.toFixed(0)}%`, 'output');
          addLine(`• VaR (95%): $${metrics.var_95.toFixed(0)}`, 'output');
          addLine('', 'output');
          addLine('[Source: BICEP path generation with real SDE integration]', 'system');
        } catch (error: any) {
          addLine(`❌ Error: ${error.message}`, 'error');
          addLine('Make sure BEF service is running: python src/backend/bef_service.py', 'system');
        }
        setIsTyping(false);
        break;

      case 'gpu':
      case 'bicep':
        setIsTyping(true);
        addLine('🚀 Initializing BICEP Monte Carlo engine...', 'output');

        try {
          const benchmark = await befService.runGPUBenchmark();
          addLine('', 'output');
          addLine('=== BICEP PERFORMANCE BENCHMARK ===', 'output');
          addLine(`Backend: ${benchmark.device}`, 'output');
          addLine('', 'output');

          for (const bench of benchmark.benchmarks) {
            addLine(`⚡ ${bench.paths.toLocaleString()} paths:`, 'output');
            addLine(`  • Time: ${bench.time_ms.toFixed(3)}ms`, 'output');
            addLine(`  • Speed: ${bench.paths_per_second.toLocaleString()} paths/sec`, 'output');
            addLine(`  • Speedup: ${bench.speedup.toFixed(1)}×`, 'output');
          }

          // Generate real paths
          const pathResult = await befService.generatePaths({
            n_paths: 1000,
            n_steps: 100,
            method: 'euler_maruyama'
          });

          addLine('', 'output');
          addLine('📈 PATH STATISTICS:', 'output');
          addLine(`• Mean: ${pathResult.statistics.mean_final.toFixed(2)}`, 'output');
          addLine(`• Std: ${pathResult.statistics.std_final.toFixed(2)}`, 'output');
          addLine(`• Max: ${pathResult.statistics.max_value.toFixed(2)}`, 'output');
          addLine(`• Min: ${pathResult.statistics.min_value.toFixed(2)}`, 'output');
        } catch (error: any) {
          addLine(`❌ Error: ${error.message}`, 'error');
        }
        setIsTyping(false);
        break;

      case 'enn':
        setIsTyping(true);
        addLine('🧠 Running ENN entangled neural network...', 'output');

        try {
          // Generate test sequence
          const sequence = Array.from({length: 20}, () => Math.random() * 2 - 1);

          const ennResult = await befService.predictWithENN({
            sequence,
            use_entanglement: true,
            lambda_param: 0.1
          });

          addLine('', 'output');
          addLine('=== ENN COMMITTOR PREDICTION ===', 'output');
          addLine(`• Committor: ${ennResult.final_prediction.toFixed(4)}`, 'output');
          addLine(`• Confidence: ${(ennResult.confidence * 100).toFixed(1)}%`, 'output');
          addLine(`• Hidden Norm: ${ennResult.hidden_state_norm.toFixed(3)}`, 'output');
          addLine(`• Entanglement λ₁: ${ennResult.entanglement_eigenvalues[0].toFixed(4)}`, 'output');
          addLine('', 'output');
          addLine('[Formula: ψₜ₊₁ = tanh(Wₓxₜ + Wₕhₜ + (E-λI)ψₜ + b)]', 'system');
        } catch (error: any) {
          addLine(`❌ Error: ${error.message}`, 'error');
        }
        setIsTyping(false);
        break;

      case 'fusion':
        setIsTyping(true);
        addLine('🎯 Running FusionAlpha planning...', 'output');

        try {
          // Generate test predictions
          const predictions = Array.from({length: 5}, () => ({
            committor: Math.random(),
            confidence: Math.random()
          }));

          const fusionResult = await befService.planWithFusionAlpha({
            enn_predictions: predictions,
            use_severity_scaling: true,
            confidence_threshold: 0.5
          });

          addLine('', 'output');
          addLine('=== FUSIONALPHA PLANNING ===', 'output');
          addLine(`• Nodes: ${fusionResult.num_nodes}`, 'output');
          addLine(`• Avg Committor: ${fusionResult.average_committor.toFixed(4)}`, 'output');
          addLine(`• Decision Distribution: ${(fusionResult.decision_distribution * 100).toFixed(1)}% BUY`, 'output');
          addLine('', 'output');

          for (const decision of fusionResult.decisions.slice(0, 3)) {
            addLine(`  ${decision.node}: ${decision.decision ? 'BUY' : 'SELL'} (conf: ${(decision.confidence * 100).toFixed(0)}%)`, 'output');
          }
        } catch (error: any) {
          addLine(`❌ Error: ${error.message}`, 'error');
        }
        setIsTyping(false);
        break;

      case 'pipeline':
        setIsTyping(true);
        addLine('🔄 Running complete BEF pipeline...', 'output');
        addLine('BICEP → ENN → FusionAlpha', 'system');

        try {
          const result = await befService.runCompletePipeline(100);

          addLine('', 'output');
          addLine('=== PIPELINE RESULTS ===', 'output');
          addLine('', 'output');
          addLine('1️⃣ BICEP:', 'output');
          addLine(`  • Generated ${result.bicep.paths_shape[0]} paths`, 'output');
          addLine(`  • Time: ${result.bicep.statistics.generation_time_ms.toFixed(2)}ms`, 'output');
          addLine('', 'output');
          addLine('2️⃣ ENN:', 'output');
          addLine(`  • Committor: ${result.enn.final_prediction.toFixed(4)}`, 'output');
          addLine(`  • Confidence: ${(result.enn.confidence * 100).toFixed(1)}%`, 'output');
          addLine('', 'output');
          addLine('3️⃣ FusionAlpha:', 'output');
          addLine(`  • Decision: ${result.fusion.decisions[0]?.decision ? 'BUY' : 'SELL'}`, 'output');
          addLine(`  • Avg Committor: ${result.fusion.average_committor.toFixed(4)}`, 'output');
        } catch (error: any) {
          addLine(`❌ Error: ${error.message}`, 'error');
        }
        setIsTyping(false);
        break;

      case 'stream':
        if (!streamConnection) {
          addLine('📡 Connecting to BEF pipeline stream...', 'system');

          befService.connectWebSocket((data) => {
            if (data.type === 'pipeline_update') {
              addLine(`[${new Date().toLocaleTimeString()}] BICEP: ${data.bicep.paths_generated} paths | ENN: ${data.enn.prediction.toFixed(3)} | Fusion: ${data.fusion.decision ? 'BUY' : 'SELL'}`, 'system');
            }
          });

          setStreamConnection(true);
          addLine('✅ Stream connected. Updates will appear here.', 'output');
        } else {
          befService.disconnectWebSocket();
          setStreamConnection(false);
          addLine('📴 Stream disconnected.', 'system');
        }
        break;

      case 'clear':
        setLines([]);
        break;

      default:
        addLine(`Command not found: ${commandName}`, 'error');
        addLine("Type 'help' for available commands", 'system');
        break;
    }
  }, [addLine, streamConnection]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (currentInput.trim()) {
      executeCommand(currentInput);
      setCurrentInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setIsPaletteOpen(true);
    }
  };

  const handlePaletteCommand = (command: string) => {
    executeCommand(command);
  };

  return (
    <div className="min-h-screen bg-terminal-bg text-terminal-text p-4 sm:p-6 md:p-8">
      <CommandPalette
        isOpen={isPaletteOpen}
        onClose={() => setIsPaletteOpen(false)}
        onCommand={handlePaletteCommand}
      />
      <div
        ref={terminalRef}
        className="max-w-5xl mx-auto font-mono text-xs sm:text-sm leading-relaxed overflow-x-auto"
      >
        <AnimatePresence>
          {lines.map((line, index) => (
            <motion.div
              key={line.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.2,
                delay: index * 0.05,
                ease: [0.4, 0, 0.2, 1]
              }}
              className={`mb-1 terminal-text ${
                line.type === 'input' ? 'text-terminal-text' :
                line.type === 'error' ? 'text-terminal-warning' :
                line.type === 'system' ? 'text-terminal-muted' :
                'text-terminal-text'
              }`}
            >
              {line.content}
            </motion.div>
          ))}
        </AnimatePresence>

        {!isTyping && (
          <motion.form
            onSubmit={handleSubmit}
            className="flex items-center mt-4 flex-wrap sm:flex-nowrap"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <span className="text-terminal-accent mr-2 text-xs sm:text-sm whitespace-nowrap">
              {INITIAL_PROMPT}
            </span>
            <input
              ref={inputRef}
              type="text"
              value={currentInput}
              onChange={(e) => setCurrentInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 bg-transparent border-none outline-none text-terminal-text font-mono text-xs sm:text-sm min-w-0"
              autoComplete="off"
              spellCheck="false"
              placeholder="Type a command..."
            />
            <span className="text-terminal-accent animate-cursor-blink ml-1 text-xs sm:text-sm">█</span>
          </motion.form>
        )}
      </div>
    </div>
  );
}