'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import CommandPalette from './CommandPalette';

interface TerminalLine {
  id: string;
  content: string | React.ReactNode;
  type: 'input' | 'output' | 'error' | 'system';
  timestamp: Date;
}

const INITIAL_PROMPT = 'ryan@rochmanofenna:~$';

export default function PortfolioTerminal() {
  const [lines, setLines] = useState<TerminalLine[]>([
    {
      id: 'welcome',
      content: (
        <div className="space-y-2">
          <div className="text-terminal-accent animate-pulse">
            Systems Engineer | Quantitative Developer | GPU Specialist
          </div>
          <div className="text-terminal-text">
            $45K funded stealth trading system | NYU CS/Math
          </div>
          <div className="text-terminal-muted mt-4">
            Type 'help' or press Cmd+K to navigate
          </div>
        </div>
      ),
      type: 'system',
      timestamp: new Date()
    }
  ]);

  const [currentInput, setCurrentInput] = useState('');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showHint, setShowHint] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const hintTimeoutRef = useRef<NodeJS.Timeout>();

  // Auto-show hint after 3 seconds of inactivity
  useEffect(() => {
    if (hintTimeoutRef.current) {
      clearTimeout(hintTimeoutRef.current);
    }

    hintTimeoutRef.current = setTimeout(() => {
      if (lines.length === 1 && !currentInput) {
        setShowHint(true);
      }
    }, 3000);

    return () => {
      if (hintTimeoutRef.current) {
        clearTimeout(hintTimeoutRef.current);
      }
    };
  }, [lines, currentInput]);

  // Command handlers
  const executeCommand = useCallback((command: string) => {
    const cmd = command.trim().toLowerCase();
    const timestamp = new Date();

    // Add input line
    setLines(prev => [...prev, {
      id: `input-${Date.now()}`,
      content: `${INITIAL_PROMPT} ${command}`,
      type: 'input',
      timestamp
    }]);

    // Add to history
    setCommandHistory(prev => [...prev, command]);
    setHistoryIndex(-1);

    // Handle commands
    switch(cmd) {
      case 'help':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: (
            <div className="space-y-1">
              <div className="text-terminal-accent mb-2">Available Commands:</div>
              <div><span className="text-terminal-success">about</span> → My story and background</div>
              <div><span className="text-terminal-success">experience</span> → Interactive work timeline</div>
              <div><span className="text-terminal-success">portfolio</span> → Live trading P&L dashboard</div>
              <div><span className="text-terminal-success">stack</span> → System architecture diagrams</div>
              <div><span className="text-terminal-success">gpu</span> → Monte Carlo GPU demo</div>
              <div><span className="text-terminal-success">skills</span> → Proven technical skills</div>
              <div><span className="text-terminal-success">projects</span> → Browse all projects</div>
              <div><span className="text-terminal-success">cv</span> → Download resume</div>
              <div><span className="text-terminal-success">clear</span> → Clear terminal</div>
            </div>
          ),
          type: 'output',
          timestamp
        }]);
        break;

      case 'about':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: (
            <TypewriterText text={`I architect systems where mathematics meets microseconds.

Currently building a funded algorithmic trading system that processes
1B market events daily with sub-20ms latency. My CUDA kernels turned
a 10-hour backtest into 1 hour.

Not your typical engineer story:
• Raised $45K non-dilutive funding at 21
• Built ML infrastructure for CFA educational content at scale
• Designed neural architectures that won NYU research competitions

I don't just implement papers—I identify where academic theory
breaks in production and engineer the bridges.

[Type 'portfolio' to see live trading metrics]
[Type 'experience' to explore my journey]`} />
          ),
          type: 'output',
          timestamp
        }]);
        break;

      case 'portfolio':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <TradingDashboard />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'experience':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <ExperienceTimeline />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'stack':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <SystemArchitecture />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'gpu':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <MonteCarloDemo />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'skills':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <SkillsProof />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'projects':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <ProjectsList />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'cv':
      case 'resume':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: (
            <div className="space-y-2">
              <div className="text-terminal-accent">Choose your format:</div>
              <div>[1] Quick scan (30 seconds) - For recruiters</div>
              <div>[2] Technical deep-dive - For engineers</div>
              <div>[3] Download PDF - Traditional format</div>
              <div className="mt-3">
                <a
                  href="/Ryan_Rochmanofenna_Resume.pdf"
                  download
                  className="text-terminal-success hover:underline"
                >
                  → Download Resume (PDF)
                </a>
              </div>
            </div>
          ),
          type: 'output',
          timestamp
        }]);
        break;

      case 'clear':
        setLines([{
          id: 'cleared',
          content: (
            <div className="text-terminal-muted">
              Terminal cleared. Type 'help' for commands.
            </div>
          ),
          type: 'system',
          timestamp
        }]);
        break;

      default:
        if (cmd) {
          setLines(prev => [...prev, {
            id: `error-${Date.now()}`,
            content: `Command not found: ${cmd}. Type 'help' for available commands.`,
            type: 'error',
            timestamp
          }]);
        }
    }

    setCurrentInput('');
  }, []);

  // Scroll to bottom when new lines added
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-[#0A0E1B] text-[#E8E9ED] p-4 sm:p-6 md:p-8">
      <div className="max-w-5xl mx-auto">
        <div
          ref={terminalRef}
          className="font-mono text-sm space-y-2 h-[80vh] overflow-y-auto custom-scrollbar"
        >
          <AnimatePresence>
            {lines.map((line) => (
              <motion.div
                key={line.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={`${
                  line.type === 'error' ? 'text-[#FF3366]' :
                  line.type === 'system' ? 'text-terminal-muted' : ''
                }`}
              >
                {line.content}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Current input line */}
          <div className="flex items-center space-x-2">
            <span className="text-[#00FF88]">{INITIAL_PROMPT}</span>
            <input
              ref={inputRef}
              type="text"
              value={currentInput}
              onChange={(e) => {
                setCurrentInput(e.target.value);
                setShowHint(false);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  executeCommand(currentInput);
                } else if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  if (historyIndex < commandHistory.length - 1) {
                    const newIndex = historyIndex + 1;
                    setHistoryIndex(newIndex);
                    setCurrentInput(commandHistory[commandHistory.length - 1 - newIndex]);
                  }
                } else if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  if (historyIndex > 0) {
                    const newIndex = historyIndex - 1;
                    setHistoryIndex(newIndex);
                    setCurrentInput(commandHistory[commandHistory.length - 1 - newIndex]);
                  } else if (historyIndex === 0) {
                    setHistoryIndex(-1);
                    setCurrentInput('');
                  }
                }
              }}
              className="flex-1 bg-transparent outline-none caret-[#00FF88]"
              autoFocus
              spellCheck={false}
            />
            <motion.span
              className="inline-block w-2 h-4 bg-[#00FF88]"
              animate={{ opacity: [1, 0] }}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
          </div>

          {/* Hint */}
          {showHint && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-terminal-muted text-xs mt-4"
            >
              Try 'portfolio' to see live trading metrics →
            </motion.div>
          )}
        </div>
      </div>

      {/* Command Palette */}
      <CommandPalette
        isOpen={showCommandPalette}
        onClose={() => setShowCommandPalette(false)}
        onCommand={(cmd) => {
          setShowCommandPalette(false);
          executeCommand(cmd);
        }}
      />
    </div>
  );
}

// Component for typewriter effect
function TypewriterText({ text }: { text: string }) {
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayedText(text.slice(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 10);
    return () => clearInterval(interval);
  }, [text]);

  return <div className="whitespace-pre-wrap">{displayedText}</div>;
}

// Trading Dashboard Component
function TradingDashboard() {
  const [pnl, setPnl] = useState(1858.95);

  useEffect(() => {
    const interval = setInterval(() => {
      setPnl(prev => prev + (Math.random() - 0.45) * 50);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="border border-[#00FF88]/30 rounded p-4 space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">
        === LIVE TRADING SYSTEM ===
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-terminal-muted text-xs">TODAY'S P&L</div>
          <div className={`text-2xl ${pnl > 0 ? 'text-[#00FF88]' : 'text-[#FF3366]'}`}>
            ${pnl.toFixed(2)} ({pnl > 0 ? '+' : ''}{(pnl/45000*100).toFixed(2)}%)
          </div>
        </div>
        <div>
          <div className="text-terminal-muted text-xs">METRICS</div>
          <div>Sharpe: 1.8 | Win Rate: 57%</div>
          <div>Max DD: 12% | Calmar: 2.4</div>
        </div>
      </div>
      <div className="text-terminal-muted text-xs">
        [Press 'B' for Monte Carlo VaR | Press 'O' for order book]
      </div>
    </div>
  );
}

// Experience Timeline Component
function ExperienceTimeline() {
  return (
    <div className="space-y-2">
      <div className="text-[#00FF88]">=== EXPERIENCE TIMELINE ===</div>
      <div>[2024-NOW] Stealth Buy-Side Research.............[ACTIVE]</div>
      <div>[2025-AUG] Sending Labs...........................[3 MOS]</div>
      <div>[2025-JUN] Video Tutor AI.........................[3 MOS]</div>
      <div>[2023-JUN] Olo....................................[4 MOS]</div>
      <div className="mt-2 text-terminal-muted">
        Type number [1-4] to expand role details
      </div>
    </div>
  );
}

// System Architecture Component
function SystemArchitecture() {
  return (
    <div className="border border-terminal-accent/30 rounded p-4">
      <pre className="text-xs">
{`┌─────────────────────────────────────┐
│        TRADING SYSTEM STACK         │
├─────────────────────────────────────┤
│  React Dashboard                    │
│       ↓                             │
│  FastAPI + WebSocket                │
│       ↓                             │
│  Kafka Event Stream                 │
│       ↓                             │
│  CUDA Compute Engine                │
│       ↓                             │
│  Order Router (FIX 4.4)            │
└─────────────────────────────────────┘

Throughput: 1B ticks/day @ p99 <20ms`}
      </pre>
    </div>
  );
}

// Monte Carlo Demo Component
function MonteCarloDemo() {
  return (
    <div className="space-y-4">
      <div className="text-[#00FF88]">=== GPU MONTE CARLO ENGINE ===</div>
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-red-500/30 rounded p-2">
          <div className="text-red-500 text-xs mb-1">CPU (Sequential)</div>
          <div>Time: 45.2s</div>
          <div className="text-xs text-terminal-muted">Single thread crawling...</div>
        </div>
        <div className="border border-[#00FF88]/30 rounded p-2">
          <div className="text-[#00FF88] text-xs mb-1">GPU (CUDA)</div>
          <div>Time: 0.003s</div>
          <div className="text-xs text-terminal-muted">15,000× faster!</div>
        </div>
      </div>
      <div>
        <div className="text-terminal-muted text-xs">RESULTS</div>
        <div>Asian Option Price: $4.2371</div>
        <div>Paths: 1,000,000 | 95% CI: [4.19, 4.28]</div>
      </div>
      <div className="text-terminal-muted text-xs">
        [Press 'C' to view CUDA kernel source]
      </div>
    </div>
  );
}

// Skills Proof Component
function SkillsProof() {
  return (
    <div className="space-y-4">
      <div className="text-[#00FF88]">=== PROVEN IN PRODUCTION ===</div>
      <div className="space-y-3">
        <div>
          <div className="text-terminal-accent">→ GPU Programming</div>
          <div className="ml-4 text-sm">
            • 10× speedup on Monte Carlo paths [see demo]<br/>
            • Custom CUDA kernels processing 1B ticks/day<br/>
            • Triton implementations with measured benchmarks
          </div>
        </div>
        <div>
          <div className="text-terminal-accent">→ Quantitative Systems</div>
          <div className="ml-4 text-sm">
            • Live trading system with $45K funding [see P&L]<br/>
            • Sharpe: 1.8 | Max Drawdown: 12% [see backtest]<br/>
            • Cointegration + mean reversion strategies deployed
          </div>
        </div>
        <div>
          <div className="text-terminal-accent">→ ML Infrastructure</div>
          <div className="ml-4 text-sm">
            • 8×V100 cluster achieving 60% faster training<br/>
            • EEG pipeline: 129 channels → 98% accuracy<br/>
            • ManimGL rendering: 5× throughput improvement
          </div>
        </div>
      </div>
      <div className="text-terminal-muted text-xs mt-4">
        Type skill name for code examples, or 'validate' for GitHub proof
      </div>
    </div>
  );
}

// Projects List Component
function ProjectsList() {
  return (
    <div className="space-y-4">
      <div className="text-[#00FF88]">=== PROJECT PORTFOLIO ===</div>
      <div className="space-y-2">
        <div className="text-terminal-accent">LIVE SYSTEMS:</div>
        <div>[1] Buy-Side Trading Stack........[RUNNING] [$45K Funded]</div>
        <div>[2] EEG Neural Pipeline...........[PUBLISHED] [NYU Greene]</div>
        <div>[3] GPU Monte Carlo Engine........[DEMO READY] [View Code]</div>

        <div className="text-terminal-accent mt-4">ARCHIVED WORK:</div>
        <div>[4] ManimGL at Sending Labs.......[COMPLETE] [5× faster]</div>
        <div>[5] Video Generation Pipeline.....[SCALED] [500+ jobs]</div>
        <div>[6] ARIMA Forecasting at Olo......[DEPLOYED] [28% RMSE↓]</div>
      </div>
      <div className="text-terminal-muted text-xs mt-4">
        Enter number or search by tech (e.g., 'cuda', 'ml')
      </div>
    </div>
  );
}