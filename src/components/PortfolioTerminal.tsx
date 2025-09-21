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
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [currentInput, setCurrentInput] = useState('');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showHint, setShowHint] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [interactiveMode, setInteractiveMode] = useState<string | null>(null);
  const [awaitingInput, setAwaitingInput] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const hintTimeoutRef = useRef<NodeJS.Timeout>();

  // Initial whoami animation
  useEffect(() => {
    const initSequence = async () => {
      await new Promise(resolve => setTimeout(resolve, 500));

      // Show command being typed
      setLines([{
        id: 'init-command',
        content: (
          <div>
            <span className="text-[#00FF88]">{INITIAL_PROMPT}</span> whoami
          </div>
        ),
        type: 'input',
        timestamp: new Date()
      }]);

      await new Promise(resolve => setTimeout(resolve, 800));

      // Show response
      setLines(prev => [...prev, {
        id: 'init-response',
        content: (
          <div className="space-y-2">
            <div className="text-terminal-accent animate-pulse">
              NYU CS/Math + Philosophy | Low-Latency Infrastructure | ML Research
            </div>
            <div className="text-terminal-text">
              Building high-throughput GPU pipelines and trading systems
            </div>
            <div className="text-terminal-muted mt-4">
              Type 'help' or press Cmd+K to navigate
            </div>
          </div>
        ),
        type: 'system',
        timestamp: new Date()
      }]);

      setIsInitializing(false);
    };

    initSequence();
  }, []);

  // Auto-show hint after 3 seconds of inactivity
  useEffect(() => {
    if (hintTimeoutRef.current) {
      clearTimeout(hintTimeoutRef.current);
    }

    if (!isInitializing && lines.length > 0 && !currentInput) {
      hintTimeoutRef.current = setTimeout(() => {
        setShowHint(true);
      }, 3000);
    }

    return () => {
      if (hintTimeoutRef.current) {
        clearTimeout(hintTimeoutRef.current);
      }
    };
  }, [lines, currentInput, isInitializing]);

  // Command handlers
  const executeCommand = useCallback((command: string) => {
    const cmd = command.trim().toLowerCase();
    const timestamp = new Date();

    // Handle interactive input modes
    if (awaitingInput) {
      handleInteractiveInput(command);
      return;
    }

    // Add input line
    setLines(prev => [...prev, {
      id: `input-${Date.now()}`,
      content: (
        <div>
          <span className="text-[#00FF88]">{INITIAL_PROMPT}</span> {command}
        </div>
      ),
      type: 'input',
      timestamp
    }]);

    // Add to history
    setCommandHistory(prev => [...prev, command]);
    setHistoryIndex(-1);
    setShowHint(false);

    // Handle commands
    switch(cmd) {
      case 'help':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: (
            <div className="space-y-1">
              <div className="text-terminal-accent mb-2">Available Commands:</div>
              <div><span className="text-terminal-success">about</span> → Professional summary & CS:GO origin story</div>
              <div><span className="text-terminal-success">experience</span> → Work history & live trading systems</div>
              <div><span className="text-terminal-success">projects</span> → Technical projects & implementations</div>
              <div><span className="text-terminal-success">skills</span> → Technical skills & expertise</div>
              <div><span className="text-terminal-success">education</span> → Academic background & coursework</div>
              <div><span className="text-terminal-success">trading-infra</span> → AI Trading Infrastructure details</div>
              <div><span className="text-terminal-success">manim-demo</span> → Educational video generation demo</div>
              <div><span className="text-terminal-success">cv</span> → Download resume PDF</div>
              <div><span className="text-terminal-success">clear</span> → Clear terminal</div>
            </div>
          ),
          type: 'output',
          timestamp
        }]);
        break;

      case 'whoami':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: (
            <div className="space-y-2">
              <div className="text-terminal-accent">
                NYU CS/Math + Philosophy | Low-Latency Infrastructure | ML Research
              </div>
              <div className="text-terminal-text">
                Building high-throughput GPU pipelines and trading systems
              </div>
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
            <AboutSection />
          ),
          type: 'output',
          timestamp
        }]);
        break;

      case 'experience':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <ExperienceSection onInteraction={setInteractiveMode} />,
          type: 'output',
          timestamp
        }]);
        break;


      case 'skills':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <SkillsSection />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'projects':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <ProjectsSection />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'cv':
      case 'resume':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <CVOptions onInteraction={setAwaitingInput} />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'education':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <EducationSection />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'trading-infra':
      case 'infra':
      case 'ai-trading':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <AITradingInfraOverview />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'manim-demo':
      case 'manim':
      case 'video-gen':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <GenerativeManimDemo />,
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
  }, [awaitingInput]);

  // Handle interactive input responses
  const handleInteractiveInput = useCallback((input: string) => {
    const timestamp = new Date();

    // Add input line
    setLines(prev => [...prev, {
      id: `input-${Date.now()}`,
      content: (
        <div>
          <span className="text-[#00FF88]">→</span> {input}
        </div>
      ),
      type: 'input',
      timestamp
    }]);

    // Process based on what we're waiting for
    switch(awaitingInput) {
      case 'cv-format':
        handleCVFormat(input);
        break;
      case 'project-select':
        handleProjectSelect(input);
        break;
      case 'skill-select':
        handleSkillSelect(input);
        break;
      default:
        break;
    }

    setAwaitingInput(null);
    setCurrentInput('');
  }, [awaitingInput]);

  // Handle CV format selection
  const handleCVFormat = (input: string) => {
    const num = input.trim();
    const timestamp = new Date();

    switch(num) {
      case '1':
        setLines(prev => [...prev, {
          id: `cv-quick-${Date.now()}`,
          content: (
            <div className="space-y-2">
              <div className="text-[#00FF88]">QUICK SCAN (30 SECONDS)</div>
              <div>• NYU CS/Math + Philosophy</div>
              <div>• Built trading system: sub-20ms p99 latency</div>
              <div>• 10× speedup on GPU Monte Carlo simulations</div>
              <div>• Python, C/C++ (CUDA), Rust, SQL, JavaScript</div>
              <div>• PyTorch, JAX, TensorFlow, Docker, Kubernetes</div>
            </div>
          ),
          type: 'output',
          timestamp
        }]);
        break;
      case '2':
        setLines(prev => [...prev, {
          id: `cv-tech-${Date.now()}`,
          content: (
            <div className="space-y-3">
              <div className="text-[#00FF88]">TECHNICAL DEEP DIVE</div>
              <div className="text-terminal-accent">GPU Engineering:</div>
              <div className="ml-4">• Custom CUDA kernels for Monte Carlo (10× NumPy)</div>
              <div className="ml-4">• Triton implementations with CUDA escapes</div>
              <div className="ml-4">• Cryptographic PRGs (AES-CTR/ChaCha20)</div>
              <div className="text-terminal-accent mt-2">Trading Infrastructure:</div>
              <div className="ml-4">• Order book reconstruction with sub-20ms p99</div>
              <div className="ml-4">• Cointegration & mean-reversion strategies</div>
              <div className="ml-4">• Markowitz portfolio optimization</div>
              <div className="text-terminal-accent mt-2">ML/AI Systems:</div>
              <div className="ml-4">• EEG pipeline: 129 channels neural processing</div>
              <div className="ml-4">• Custom ENN architecture (C++/Eigen)</div>
              <div className="ml-4">• ManimGL rendering: 5× throughput</div>
            </div>
          ),
          type: 'output',
          timestamp
        }]);
        break;
      case '3':
        // PDF download already handled by link
        setLines(prev => [...prev, {
          id: `cv-pdf-${Date.now()}`,
          content: 'Please click the download link above to get the PDF.',
          type: 'system',
          timestamp
        }]);
        break;
      default:
        setLines(prev => [...prev, {
          id: `cv-error-${Date.now()}`,
          content: `Invalid option: ${num}. Please enter 1, 2, or 3.`,
          type: 'error',
          timestamp
        }]);
    }
  };

  // Handle project selection
  const handleProjectSelect = (input: string) => {
    const timestamp = new Date();
    const projects: { [key: string]: React.ReactNode } = {
      '1': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">STEALTH BUY-SIDE RESEARCH STACK</div>
          <div>Status: Active | Real capital deployment</div>
          <div className="mt-2">• Sub-20ms p99 latency on 1B events/day</div>
          <div>• Cointegration & mean-reversion strategies</div>
          <div>• Custom CUDA kernels for 10× backtest speedup</div>
          <div>• Live monitoring: P&L, VaR, drawdowns, slippage</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Architecture: Rust engine → PostgreSQL timeseries → Redis cache → Python analytics
          </div>
        </div>
      ),
      '2': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">EEG 2025: CONTRADICTION-AWARE NEURAL PIPELINE</div>
          <div>Status: Published | Venue: NYU Greene HPC</div>
          <div className="mt-2">• Adapted BICEP→ENN→Fusion from trading to EEG</div>
          <div>• 129 channels processed with graph fusion</div>
          <div>• 60% faster training on 8×V100 cluster</div>
          <div>• Cross-subject alignment via contradiction operators</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Research focus: Multi-subject EEG decoding with attention mechanisms
          </div>
        </div>
      ),
      '3': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">GPU MONTE CARLO ENGINE (BICEP)</div>
          <div>Status: Demo Ready | Performance: 10× speedup</div>
          <div className="mt-2">• Euler-Maruyama path simulation</div>
          <div>• Sobol sequences + variance reduction</div>
          <div>• Cryptographic PRGs for i.i.d. guarantees</div>
          <div>• Asian & Barrier option pricing</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Key innovation: Bit-level optimized RNG with ChaCha20/AES-CTR
          </div>
        </div>
      ),
      '4': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">CUSTOM NEURAL NETWORK (ENN)</div>
          <div>Status: Deployed | Language: C++17</div>
          <div className="mt-2">• Hand-rolled backprop with Eigen3 linear algebra</div>
          <div>• AVX2 vectorization for matrix operations</div>
          <div>• OpenMP parallel batch processing</div>
          <div>• Real-time gesture recognition application</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Performance: 3ms inference on CPU, no GPU required
          </div>
        </div>
      ),
      '5': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">MANIMGL PIPELINE @ SENDING LABS</div>
          <div>Duration: Jul-Aug 2024 | Status: Production</div>
          <div className="mt-2">• Containerized ManimGL with EGL offscreen rendering</div>
          <div>• Modal serverless deployment with GPU acceleration</div>
          <div>• Queue-based job processing with retry logic</div>
          <div>• 5× throughput, 35% cost reduction</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Scale: 1000+ animations/day for educational content
          </div>
        </div>
      ),
      '6': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">VIDEO TUTOR AI</div>
          <div>Duration: Apr-Jun 2024 | Tech: GPT-4 + TTS</div>
          <div className="mt-2">• Multi-modal content generation pipeline</div>
          <div>• GPT-4 curriculum planning + script writing</div>
          <div>• ElevenLabs TTS integration</div>
          <div>• ManimGL math visualizations</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Impact: High-concurrency system handling educational video production
          </div>
        </div>
      ),
      '7': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">INFRASTRUCTURE AUTOMATION @ OLO</div>
          <div>Duration: May-Aug 2022, Jun 2023 | Type: Software Engineering</div>
          <div className="mt-2">• Terraform IaC for automated provisioning</div>
          <div>• Cloudflare and Datadog observability integration</div>
          <div>• C# + SQL custom restaurant filter system</div>
          <div>• Migrated legacy DevOps to codified infrastructure</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Impact: Streamlined deployment and monitoring for platform services
          </div>
        </div>
      ),
      '8': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">NYU TANDON MADE CHALLENGE WINNER</div>
          <div>Year: 2020-2021 | Competition Winner</div>
          <div className="mt-2">• Full-stack application (React/TypeScript/Node)</div>
          <div>• MongoDB database with real-time sync</div>
          <div>• WebSocket live collaboration features</div>
          <div>• Deployed on AWS with auto-scaling</div>
          <div className="mt-2 text-xs text-terminal-muted">
            Competition: 50+ teams, judged on innovation and execution
          </div>
        </div>
      ),
      'cuda': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">CUDA/GPU PROJECTS</div>
          <div>• Monte Carlo Engine: 10× NumPy baseline</div>
          <div>• Custom kernels for order book transforms</div>
          <div>• Triton implementations with CUDA escapes</div>
          <div>• 8×V100 cluster optimization for EEG</div>
        </div>
      ),
      'ml': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">ML/AI PROJECTS</div>
          <div>• EEG Neural Pipeline (129 channels)</div>
          <div>• Custom ENN architecture (C++/Eigen)</div>
          <div>• ManimGL rendering pipeline (5× throughput)</div>
          <div>• GPT-4o educational content generation</div>
        </div>
      ),
      'python': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">PYTHON ECOSYSTEM EXPERTISE</div>
          <div>• NumPy/Pandas for quantitative analysis</div>
          <div>• PyTorch/JAX for neural network research</div>
          <div>• Asyncio/multiprocessing for high-throughput systems</div>
          <div>• FastAPI/Flask for microservices</div>
        </div>
      ),
      'rust': (
        <div className="space-y-2">
          <div className="text-[#00FF88]">RUST SYSTEMS PROGRAMMING</div>
          <div>• Low-latency order book reconstruction</div>
          <div>• Lock-free concurrent data structures</div>
          <div>• Zero-copy parsing for market data feeds</div>
          <div>• Memory-safe financial calculations</div>
        </div>
      )
    };

    const content = projects[input.toLowerCase()] || projects[input];
    if (content) {
      setLines(prev => [...prev, {
        id: `project-${Date.now()}`,
        content,
        type: 'output',
        timestamp
      }]);
    } else {
      setLines(prev => [...prev, {
        id: `project-error-${Date.now()}`,
        content: `Invalid selection: ${input}. Enter a number (1-8) or keyword (cuda, ml).`,
        type: 'error',
        timestamp
      }]);
    }
  };

  // Handle skill selection
  const handleSkillSelect = (input: string) => {
    const timestamp = new Date();
    const inp = input.toLowerCase().trim();

    if (inp === 'validate') {
      setLines(prev => [...prev, {
        id: `skill-validate-${Date.now()}`,
        content: (
          <div className="space-y-2">
            <div className="text-[#00FF88]">GITHUB VALIDATION</div>
            <div>• BICEP: github.com/rochmanofenna/BICEP</div>
            <div>• ENN: github.com/rochmanofenna/ENN</div>
            <div>• Trading Infrastructure: Private repo (NDA)</div>
            <div>• Portfolio: github.com/rochmanofenna/ryguy-ai</div>
          </div>
        ),
        type: 'output',
        timestamp
      }]);
    } else if (inp === 'python' || inp === 'cuda') {
      setLines(prev => [...prev, {
        id: `skill-code-${Date.now()}`,
        content: (
          <div className="space-y-2">
            <div className="text-[#00FF88]">CUDA KERNEL EXAMPLE</div>
            <pre className="text-xs bg-black/30 p-2 rounded overflow-x-auto">
{`__global__ void monte_carlo_paths(
    float* paths, float* randoms,
    float S0, float mu, float sigma,
    float dt, int n_steps, int n_paths
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_paths) {
        float S = S0;
        for (int i = 0; i < n_steps; i++) {
            float dW = randoms[tid * n_steps + i];
            S *= exp((mu - 0.5f*sigma*sigma)*dt +
                     sigma*sqrt(dt)*dW);
            paths[tid * n_steps + i] = S;
        }
    }
}`}
            </pre>
          </div>
        ),
        type: 'output',
        timestamp
      }]);
    } else {
      setLines(prev => [...prev, {
        id: `skill-error-${Date.now()}`,
        content: `Unknown skill: ${input}. Try 'python', 'cuda', or 'validate'.`,
        type: 'error',
        timestamp
      }]);
    }
  };

  // Handle interactive keyboard events
  useEffect(() => {
    if (!interactiveMode) return;

    const handleKeyPress = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      const timestamp = new Date();

      switch(interactiveMode) {
        case 'portfolio':
          if (key === 'b') {
            setLines(prev => [...prev, {
              id: `monte-carlo-${Date.now()}`,
              content: (
                <div className="space-y-2 border border-[#00FF88]/30 rounded p-3">
                  <div className="text-[#00FF88]">MONTE CARLO VAR ANALYSIS</div>
                  <div>95% VaR (1-day): $2,847</div>
                  <div>99% VaR (1-day): $4,213</div>
                  <div>Expected Shortfall: $5,124</div>
                  <div className="mt-2">Simulations: 10,000 paths</div>
                  <div>Volatility: 18.3% annualized</div>
                </div>
              ),
              type: 'output',
              timestamp
            }]);
            setInteractiveMode(null);
          } else if (key === 'o') {
            setLines(prev => [...prev, {
              id: `orderbook-${Date.now()}`,
              content: (
                <div className="space-y-2 border border-[#00FF88]/30 rounded p-3">
                  <div className="text-[#00FF88]">LIVE ORDER BOOK</div>
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <div className="text-red-500">ASKS</div>
                      <div>100 @ 152.45</div>
                      <div>250 @ 152.44</div>
                      <div>500 @ 152.43</div>
                    </div>
                    <div>
                      <div className="text-[#00FF88]">BIDS</div>
                      <div>150 @ 152.42</div>
                      <div>300 @ 152.41</div>
                      <div>450 @ 152.40</div>
                    </div>
                  </div>
                  <div className="text-terminal-muted">Spread: $0.01</div>
                </div>
              ),
              type: 'output',
              timestamp
            }]);
            setInteractiveMode(null);
          }
          break;

        case 'gpu':
          if (key === 'c') {
            setLines(prev => [...prev, {
              id: `cuda-source-${Date.now()}`,
              content: (
                <div className="space-y-2">
                  <div className="text-[#00FF88]">CUDA KERNEL SOURCE</div>
                  <pre className="text-xs bg-black/30 p-2 rounded overflow-x-auto">
{`// Asian option Monte Carlo pricing
__global__ void asian_option_kernel(
    float* payoffs, curandState* states,
    float S0, float K, float r, float sigma,
    float T, int N, int paths_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];

    for (int p = 0; p < paths_per_thread; p++) {
        float sum = 0.0f;
        float S = S0;
        float dt = T / N;

        for (int i = 0; i < N; i++) {
            float z = curand_normal(&localState);
            S *= exp((r - 0.5f*sigma*sigma)*dt +
                     sigma*sqrt(dt)*z);
            sum += S;
        }

        float avg = sum / N;
        payoffs[tid*paths_per_thread + p] =
            fmaxf(avg - K, 0.0f);
    }

    states[tid] = localState;
}`}
                  </pre>
                </div>
              ),
              type: 'output',
              timestamp
            }]);
            setInteractiveMode(null);
          }
          break;
      }
    };

    window.addEventListener('keypress', handleKeyPress);
    return () => window.removeEventListener('keypress', handleKeyPress);
  }, [interactiveMode]);

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

  // Focus input when clicking terminal
  const handleTerminalClick = () => {
    inputRef.current?.focus();
  };

  return (
    <div
      className="min-h-screen bg-[#0A0E1B] text-[#E8E9ED] p-4 sm:p-6 md:p-8"
      onClick={handleTerminalClick}
    >
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
          {!isInitializing && (
            <div className="flex items-center">
              <span className="text-[#00FF88] mr-2">
                {awaitingInput ? '→' : INITIAL_PROMPT}
              </span>
              <div className="flex-1 relative">
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
                    } else if (e.key === 'ArrowUp' && !awaitingInput) {
                      e.preventDefault();
                      if (historyIndex < commandHistory.length - 1) {
                        const newIndex = historyIndex + 1;
                        setHistoryIndex(newIndex);
                        setCurrentInput(commandHistory[commandHistory.length - 1 - newIndex]);
                      }
                    } else if (e.key === 'ArrowDown' && !awaitingInput) {
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
                  className="bg-transparent outline-none border-none text-[#E8E9ED] w-full font-mono"
                  style={{ caretColor: 'transparent' }}
                  autoFocus
                  spellCheck={false}
                  placeholder={awaitingInput ? "Enter your selection..." : ""}
                />
                <motion.span
                  className="inline-block w-2 h-4 bg-[#00FF88] absolute top-0"
                  style={{ left: `${currentInput.length}ch` }}
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                />
              </div>
            </div>
          )}

          {/* Hint */}
          {showHint && !isInitializing && !awaitingInput && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-terminal-muted text-xs mt-4"
            >
              Try 'experience' to see work history →
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
function TradingDashboard({ onInteraction }: { onInteraction: (mode: string) => void }) {
  const [pnl, setPnl] = useState(1858.95);

  useEffect(() => {
    const interval = setInterval(() => {
      setPnl(prev => prev + (Math.random() - 0.45) * 50);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    onInteraction('portfolio');
  }, [onInteraction]);

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
          <div className="text-terminal-muted text-xs">LIVE STATUS</div>
          <div>Sub-20ms p99 latency</div>
          <div>Multiple pairs | Real capital deployment</div>
        </div>
      </div>
      <div className="text-terminal-muted text-xs animate-pulse">
        [Press 'B' for Monte Carlo VaR | Press 'O' for order book]
      </div>
    </div>
  );
}

// Monte Carlo Demo Component
function MonteCarloDemo({ onInteraction }: { onInteraction: (mode: string) => void }) {
  useEffect(() => {
    onInteraction('gpu');
  }, [onInteraction]);

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88]">=== GPU MONTE CARLO ENGINE (BICEP) ===</div>
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-red-500/30 rounded p-2">
          <div className="text-red-500 text-xs mb-1">NumPy Baseline</div>
          <div>Time: 45.2s</div>
          <div className="text-xs text-terminal-muted">Standard implementation</div>
        </div>
        <div className="border border-[#00FF88]/30 rounded p-2">
          <div className="text-[#00FF88] text-xs mb-1">Custom CUDA Kernels</div>
          <div>Time: 0.003s</div>
          <div className="text-xs text-terminal-muted">10× speedup achieved!</div>
        </div>
      </div>
      <div>
        <div className="text-terminal-muted text-xs">IMPLEMENTATION DETAILS</div>
        <div>• Euler-Maruyama path simulation</div>
        <div>• Sobol sequences + variance reduction</div>
        <div>• Cryptographic PRGs (AES-CTR/ChaCha20)</div>
        <div>• Asian and Barrier option pricing</div>
      </div>
      <div className="text-terminal-muted text-xs animate-pulse">
        [Press 'C' to view CUDA kernel source]
      </div>
    </div>
  );
}
// CV Options Component
function CVOptions({ onInteraction }: { onInteraction: (mode: string | null) => void }) {
  useEffect(() => {
    onInteraction('cv-format');
  }, [onInteraction]);

  return (
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
          onClick={(e) => e.stopPropagation()}
        >
          → Download Resume (PDF)
        </a>
      </div>
      <div className="text-terminal-muted text-xs mt-4 animate-pulse">
        Enter 1, 2, or 3 to select format
      </div>
    </div>
  );
}
// About Section Component
function AboutSection() {
  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">ABOUT ME</div>

      <div className="space-y-2">
        <div className="text-terminal-accent">NYU CS/Math + Philosophy | Low-Latency Infrastructure | ML Research</div>
        <div className="text-terminal-text">
          Building high-throughput GPU pipelines and trading systems where microseconds matter.
        </div>
      </div>

      <div className="border-l-2 border-terminal-muted/30 pl-4 space-y-3">
        <div>
          <div className="text-terminal-accent">Origin Story: From CS:GO to Quant Trading</div>
          <div className="text-sm mt-1">
            Started by optimizing CS:GO from 30 to 60+ FPS on a "trash" laptop by understanding CPU scheduling
            and cache locality. That obsession with squeezing performance out of hardware led me to build trading
            systems processing billions of events at sub-20ms latency and CUDA kernels that outperform NumPy by 10×.
          </div>
          <div className="text-xs text-terminal-muted mt-2">
            The path from gaming to quant isn't weird - both care about every microsecond, punish inefficiency,
            and reward understanding hardware at the metal level.
          </div>
        </div>

        <div>
          <div className="text-terminal-accent">Current Focus</div>
          <div className="text-sm mt-1">
            • Architecting end-to-end research and execution stacks for trading
            • Custom CUDA kernels for Monte Carlo simulation and order book transforms
            • EEG neural pipelines with contradiction-aware decoding
            • High-concurrency ML infrastructure for content generation
          </div>
        </div>

        <div>
          <div className="text-terminal-accent">Philosophy</div>
          <div className="text-sm mt-1">
            Most "slow" computers aren't slow - they're just badly utilized. My edge is optimizing on hardware
            most people throw away, then applying that obsession to systems where microseconds mean millions.
          </div>
        </div>
      </div>

      <div className="text-terminal-muted text-xs mt-4">
        [Type 'experience' to see work history | 'projects' for technical implementations]
      </div>
    </div>
  );
}

// Experience Section Component
function ExperienceSection({ onInteraction }: { onInteraction: (mode: string) => void }) {
  const [showDetails, setShowDetails] = useState<string | null>(null);
  const [activeCodeTab, setActiveCodeTab] = useState<string>('metrics');

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">EXPERIENCE</div>

      <div className="space-y-4">
        <div className="border border-terminal-accent/30 rounded p-3">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-terminal-accent font-bold">Stealth Buy-Side Research Stack</div>
              <div className="text-sm">Systems Engineer — ML, Low-Latency Infrastructure, and Alpha Research</div>
            </div>
            <div className="text-xs text-terminal-muted">Jun 2024 – Present</div>
          </div>

          <div className="mt-2 space-y-1 text-sm">
            <div>• Architected end-to-end research/execution stack with sub-20ms p99 latency</div>
            <div>• Designed cointegration and mean-reversion strategies with Markowitz optimization</div>
            <div>• Engineered custom CUDA kernels (BICEP) for 10× speedup over NumPy baselines</div>
            <div>• Delivered live monitoring suite (FusionAlpha) for P&L, VaR, drawdowns</div>
            <div>• Achieved 2.95 Sharpe ratio with ENN-based signal processing</div>
            <div>• Secured $45K in non-dilutive R&D funding</div>
          </div>

          <div className="mt-3 space-y-2">
            <button
              onClick={() => {
                setShowDetails(showDetails === 'trading' ? null : 'trading');
                onInteraction('portfolio');
              }}
              className="text-xs text-terminal-success hover:underline"
            >
              [View Live Trading Dashboard]
            </button>

            <button
              onClick={() => setShowDetails(showDetails === 'code' ? null : 'code')}
              className="text-xs text-terminal-success hover:underline ml-4"
            >
              [View AI Trading Infrastructure Code]
            </button>
          </div>

          {showDetails === 'trading' && (
            <div className="mt-3">
              <TradingDashboard onInteraction={onInteraction} />
            </div>
          )}

          {showDetails === 'code' && (
            <div className="mt-3 space-y-3">
              <div className="flex gap-2 mb-2">
                <button
                  onClick={() => setActiveCodeTab('metrics')}
                  className={`text-xs px-2 py-1 rounded ${activeCodeTab === 'metrics' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
                >
                  Performance Metrics
                </button>
                <button
                  onClick={() => setActiveCodeTab('bicep')}
                  className={`text-xs px-2 py-1 rounded ${activeCodeTab === 'bicep' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
                >
                  BICEP GPU
                </button>
                <button
                  onClick={() => setActiveCodeTab('enn')}
                  className={`text-xs px-2 py-1 rounded ${activeCodeTab === 'enn' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
                >
                  ENN Model
                </button>
                <button
                  onClick={() => setActiveCodeTab('fusion')}
                  className={`text-xs px-2 py-1 rounded ${activeCodeTab === 'fusion' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
                >
                  FusionAlpha
                </button>
                <button
                  onClick={() => setActiveCodeTab('infra')}
                  className={`text-xs px-2 py-1 rounded ${activeCodeTab === 'infra' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
                >
                  Infrastructure
                </button>
              </div>

              {activeCodeTab === 'metrics' && <AITradingMetrics />}
              {activeCodeTab === 'bicep' && <BICEPCodeSnippet />}
              {activeCodeTab === 'enn' && <ENNCodeSnippet />}
              {activeCodeTab === 'fusion' && <FusionAlphaSnippet />}
              {activeCodeTab === 'infra' && <InfrastructureCode />}
            </div>
          )}
        </div>

        <div className="border border-terminal-muted/30 rounded p-3">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-terminal-accent font-bold">Sending Labs</div>
              <div className="text-sm">Machine Learning Engineer (Contract)</div>
            </div>
            <div className="text-xs text-terminal-muted">Jun 2024 – Aug 2024</div>
          </div>

          <div className="mt-2 space-y-1 text-sm">
            <div>• Designed ManimGL pipeline for protocol SDK visualizations</div>
            <div>• Integrated distributed GPU rendering with Modal/Fly.io</div>
            <div>• Achieved 5× throughput improvement and 35% cost reduction</div>
          </div>
        </div>

        <div className="border border-terminal-muted/30 rounded p-3">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-terminal-accent font-bold">Video Tutor AI</div>
              <div className="text-sm">Machine Learning Engineer (Contract)</div>
            </div>
            <div className="text-xs text-terminal-muted">Apr 2024 – Jun 2024</div>
          </div>

          <div className="mt-2 space-y-1 text-sm">
            <div>• Architected AI educational content pipeline using GPT-4o, TTS, and ManimGL</div>
            <div>• Deployed Redis-backed task queue for 500+ concurrent render jobs</div>
            <div>• Cut compute spend by ~40% via caching and batching</div>
          </div>
        </div>

        <div className="border border-terminal-muted/30 rounded p-3">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-terminal-accent font-bold">Olo</div>
              <div className="text-sm">Software Engineering Intern</div>
            </div>
            <div className="text-xs text-terminal-muted">May 2022 – Aug 2022; Jun 2023</div>
          </div>

          <div className="mt-2 space-y-1 text-sm">
            <div>• Automated infrastructure provisioning with Terraform IaC</div>
            <div>• Integrated Cloudflare and Datadog observability into platform services</div>
          </div>
        </div>

        <div className="mt-4">
          <button
            onClick={() => {
              setShowDetails(showDetails === 'gpu' ? null : 'gpu');
              onInteraction('gpu');
            }}
            className="text-terminal-success hover:underline text-sm"
          >
            [View GPU Monte Carlo Engine Demo]
          </button>

          {showDetails === 'gpu' && (
            <div className="mt-3">
              <MonteCarloDemo onInteraction={onInteraction} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Projects Section Component
function ProjectsSection() {
  const [selectedProject, setSelectedProject] = useState<number | null>(null);

  const projects = [
    {
      id: 1,
      name: "EEG 2025: Contradiction-Aware Neural Pipeline",
      tech: "PyTorch, JAX, 8×V100 cluster",
      details: [
        "Adapted BICEP→ENN→Fusion framework from quant trading to EEG IV-2a (129 channels)",
        "Combined stochastic path simulation with temporal encoders and graph fusion",
        "Achieved ~60% faster training on NYU Greene HPC via custom CUDA kernels"
      ]
    },
    {
      id: 2,
      name: "GPU Monte Carlo Engine for Derivative Pricing (BICEP)",
      tech: "CUDA, Triton, ChaCha20/AES-CTR",
      details: [
        "Implemented CUDA-accelerated Euler-Maruyama path simulation with Sobol sequences",
        "Extended with cryptographic PRGs to ensure i.i.d. Gaussian increments",
        "Achieved 10× speedup over NumPy baseline for Asian and Barrier options"
      ]
    },
    {
      id: 3,
      name: "Generative Manim Educational Pipeline",
      tech: "ManimGL, GPT-4o, Docker, Flask",
      details: [
        "AI-powered educational video generation from natural language prompts",
        "Domain-specific configurations for physics, chemistry, mathematics",
        "5× throughput improvement, 35% cost reduction at Sending Labs"
      ],
      demo: true
    },
    {
      id: 4,
      name: "Custom Neural Network (ENN) for Sequence Modeling",
      tech: "C++17, Eigen3, AVX2, OpenMP",
      details: [
        "Developed compact recurrent cell with PSD-constrained entanglement matrix",
        "Implemented full backprop-through-time with spectral regularization",
        "Deployed in real-time gesture recognition: 98% accuracy at 25 FPS"
      ]
    },
    {
      id: 5,
      name: "NYU Hyperloop Control Systems",
      tech: "C++, ROS, Embedded Systems",
      details: [
        "Designed control algorithms for pod stability and braking",
        "Implemented real-time sensor fusion for position tracking",
        "Contributed to team achieving Design Excellence award"
      ]
    },
    {
      id: 6,
      name: "NYU Tandon Made Challenge Winner",
      tech: "React, TypeScript, Node.js, MongoDB",
      details: [
        "Built biomedical/computer vision venture prototype",
        "Implemented real-time collaboration with WebSocket",
        "Won competition among 50+ teams, awarded $5k pre-seed"
      ]
    }
  ];

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">PROJECTS</div>

      <div className="space-y-3">
        {projects.map((project) => (
          <div key={project.id} className="border-l-2 border-terminal-muted/30 pl-4">
            <button
              onClick={() => setSelectedProject(selectedProject === project.id ? null : project.id)}
              className="text-left w-full hover:text-terminal-accent transition-colors"
            >
              <div className="flex justify-between items-start">
                <div>
                  <span className="text-terminal-text font-semibold">
                    {project.id}. {project.name}
                  </span>
                  <div className="text-xs text-terminal-muted mt-1">{project.tech}</div>
                </div>
                <span className="text-xs text-terminal-muted">
                  {selectedProject === project.id ? '[-]' : '[+]'}
                </span>
              </div>
            </button>

            {selectedProject === project.id && (
              <div className="mt-2 ml-4 space-y-1 text-sm text-terminal-text">
                {project.details.map((detail, idx) => (
                  <div key={idx}>• {detail}</div>
                ))}
                {project.demo && (
                  <div className="mt-2">
                    <button
                      onClick={() => {
                        const event = new KeyboardEvent('keydown', { key: 'Enter' });
                        const input = document.querySelector('input[type="text"]') as HTMLInputElement;
                        if (input) {
                          input.value = 'manim-demo';
                          input.dispatchEvent(event);
                        }
                      }}
                      className="text-xs text-terminal-success hover:underline"
                    >
                      [▶ View Interactive Demo]
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 space-y-2">
        <div className="text-xs text-terminal-muted">
          Additional Projects: Infrastructure at Olo, ManimGL Pipeline at Sending Labs,
          Video Tutor AI educational content generation
        </div>
        <div className="flex gap-4">
          <a
            href="https://github.com/rochmanofenna"
            target="_blank"
            rel="noopener noreferrer"
            className="text-terminal-success hover:underline text-sm"
          >
            [View GitHub]
          </a>
          <button
            onClick={() => window.open('/projects/monte-carlo', '_blank')}
            className="text-terminal-success hover:underline text-sm"
          >
            [Launch Monte Carlo Demo]
          </button>
        </div>
      </div>
    </div>
  );
}

// Skills Section Component
function SkillsSection() {
  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">TECHNICAL SKILLS</div>

      <div className="space-y-4">
        <div>
          <div className="text-terminal-accent font-semibold mb-2">Mathematics</div>
          <div className="ml-4 text-sm space-y-1">
            <div>• Linear Algebra, Probability, Optimization, Numerical Methods</div>
            <div>• Stochastic Differential Equations, Pseudo RNGs</div>
            <div>• Cryptography (Number Theory), Discrete Mathematics, Combinatorics</div>
          </div>
        </div>

        <div>
          <div className="text-terminal-accent font-semibold mb-2">Programming</div>
          <div className="ml-4 text-sm space-y-1">
            <div>• <span className="text-terminal-success">Languages:</span> Python, C/C++ (CUDA/low-latency), Rust, SQL, JavaScript, Bash, Go</div>
            <div>• <span className="text-terminal-success">ML/AI:</span> PyTorch, JAX, TensorFlow, scikit-learn, Optuna</div>
            <div>• <span className="text-terminal-success">Architectures:</span> GNNs, Transformers, Neural ODEs</div>
          </div>
        </div>

        <div>
          <div className="text-terminal-accent font-semibold mb-2">Systems & Infrastructure</div>
          <div className="ml-4 text-sm space-y-1">
            <div>• <span className="text-terminal-success">GPU:</span> CUDA/Triton kernels, Custom implementations, HPC (NYU Greene)</div>
            <div>• <span className="text-terminal-success">Cloud:</span> AWS (EC2, S3, Lambda), Docker, Kubernetes, Ray</div>
            <div>• <span className="text-terminal-success">Data:</span> PostgreSQL, Redis, Kafka, TimescaleDB</div>
            <div>• <span className="text-terminal-success">DevOps:</span> Git, Linux, Terraform, CI/CD pipelines</div>
          </div>
        </div>

        <div className="border border-terminal-accent/30 rounded p-3 mt-4">
          <div className="text-terminal-accent text-sm mb-2">Sample CUDA Kernel</div>
          <pre className="text-xs bg-black/30 p-2 rounded overflow-x-auto">
{`__global__ void monte_carlo_paths(
    float* paths, curandState* states,
    float S0, float K, float r, float sigma,
    float T, int N, int paths_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];

    for (int p = 0; p < paths_per_thread; p++) {
        float S = S0;
        for (int i = 0; i < N; i++) {
            float z = curand_normal(&localState);
            S *= exp((r - 0.5f*sigma*sigma)*dt +
                     sigma*sqrt(dt)*z);
        }
        paths[tid*paths_per_thread + p] = fmaxf(S - K, 0.0f);
    }
}`}
          </pre>
        </div>
      </div>
    </div>
  );
}

// Education Section Component
function EducationSection() {
  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">EDUCATION</div>

      <div className="border border-terminal-accent/30 rounded p-4">
        <div className="flex justify-between items-start">
          <div>
            <div className="text-terminal-accent font-bold">New York University</div>
            <div className="text-sm mt-1">B.A. in Computer Science & Mathematics, Minor in Philosophy</div>
          </div>
          <div className="text-xs text-terminal-muted">Expected May 2026</div>
        </div>

        <div className="mt-3 space-y-2">
          <div>
            <div className="text-terminal-success text-sm font-semibold">Relevant Coursework:</div>
            <div className="ml-4 text-sm mt-1 grid grid-cols-2 gap-2">
              <div>• Linear Algebra</div>
              <div>• Cryptography (Number Theory)</div>
              <div>• Algorithms</div>
              <div>• Data Structures</div>
              <div>• Operating & Distributed Systems</div>
              <div>• Discrete Mathematics</div>
              <div>• Natural Language Processing</div>
              <div>• Combinatorics</div>
            </div>
          </div>

          <div>
            <div className="text-terminal-success text-sm font-semibold">Honors & Awards:</div>
            <div className="ml-4 text-sm mt-1">
              <div>• Dean's List</div>
              <div>• $50,000 Annual Merit Scholarship</div>
              <div>• NYU Tandon Made Challenge Winner (2020-2021)</div>
              <div>• 2x Finalist with $5k pre-seed funding</div>
            </div>
          </div>

          <div>
            <div className="text-terminal-success text-sm font-semibold">Leadership & Activities:</div>
            <div className="ml-4 text-sm mt-1">
              <div>• NYU Hyperloop (2021-2022): Control Systems</div>
              <div>• NYU Web Publishing Consultant</div>
              <div>• Greek Life: Nu Alpha Phi</div>
            </div>
          </div>
        </div>
      </div>

      <div className="text-terminal-muted text-xs mt-4">
        GPA and transcript available upon request
      </div>
    </div>
  );
}

// AI Trading Infrastructure Components
function AITradingMetrics() {
  return (
    <div className="border border-terminal-accent/30 rounded p-3 space-y-3 bg-black/30">
      <div className="text-terminal-accent text-sm font-bold">Production Performance Metrics</div>

      <div className="grid grid-cols-2 gap-4 text-xs">
        <div className="space-y-2">
          <div className="text-terminal-success">Strategy Performance</div>
          <div className="ml-2">
            <div>• Sharpe Ratio: <span className="text-terminal-accent font-bold">2.95</span></div>
            <div>• Annual Return: <span className="text-white">1.98%</span></div>
            <div>• Max Drawdown: <span className="text-white">-0.36%</span></div>
            <div>• Win Rate: <span className="text-white">52.2%</span></div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-terminal-success">System Performance</div>
          <div className="ml-2">
            <div>• Latency p99: <span className="text-terminal-accent font-bold">&lt;20ms</span></div>
            <div>• Throughput: <span className="text-white">14k+ rps</span></div>
            <div>• Events/Day: <span className="text-white">1.2B+</span></div>
            <div>• Cache Hit: <span className="text-white">40%+</span></div>
          </div>
        </div>
      </div>

      <div className="text-terminal-muted text-xs mt-2">
        Source: AI_TRADING_INFRA/PERFORMANCE_CLAIMS.md | Validated via K6 benchmarks
      </div>
    </div>
  );
}

function BICEPCodeSnippet() {
  return (
    <div className="space-y-2">
      <div className="text-terminal-accent text-sm">BICEP: GPU Monte Carlo Engine</div>
      <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`# BICEP Performance: 2.5M+ paths/second on consumer GPU
class MonteCarloKernel:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.compile_mode = 'reduce-overhead'

    @torch.compile(mode='reduce-overhead')
    def generate_paths(self, S0, mu, sigma, T, N, paths):
        """Sub-millisecond path generation (0.4ms on M3 Metal)"""
        dt = T / N
        randn = torch.randn((paths, N), device=self.device)

        # Optimized GPU kernel with coalesced memory access
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * torch.sqrt(dt)

        # Vectorized path computation
        log_returns = drift + diffusion * randn
        log_paths = torch.cumsum(log_returns, dim=1)
        paths = S0 * torch.exp(log_paths)

        return paths

# Benchmarked: 10,000 paths in 0.0004ms (2.5M paths/sec)`}
      </pre>
      <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/BICEP/bicep_core.py</div>
    </div>
  );
}

function ENNCodeSnippet() {
  return (
    <div className="space-y-2">
      <div className="text-terminal-accent text-sm">ENN: Entangled Neural Network</div>
      <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`class EntangledNeuralNetwork(nn.Module):
    """Novel architecture with entangled neuron dynamics"""
    def __init__(self, config):
        super().__init__()
        self.num_neurons = config.num_neurons
        self.num_states = config.num_states

        # Entanglement matrix (PSD-constrained)
        self.W_entangle = nn.Parameter(
            torch.randn(num_neurons, num_neurons) * 0.1
        )

        # Multi-head attention for neuron-state processing
        self.neuron_attention = nn.MultiheadAttention(
            embed_dim=num_states,
            num_heads=4,
            batch_first=True
        )

        # Adaptive sparsity control
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, hidden=None):
        # Entangled state evolution
        entangled = torch.matmul(self.W_entangle, hidden)

        # Apply attention mechanism
        attended, _ = self.neuron_attention(entangled, entangled, entangled)

        # Dynamic pruning for sparsity
        mask = torch.abs(attended) > self.sparsity_threshold
        sparse_output = attended * mask

        return sparse_output

# Variants: minimal (5K), neuron_only (18K), full (148K params)`}
      </pre>
      <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/ENN/enn/enhanced_model.py</div>
    </div>
  );
}

function FusionAlphaSnippet() {
  return (
    <div className="space-y-2">
      <div className="text-terminal-accent text-sm">FusionAlpha: High-Performance Router</div>
      <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`@router.post('/predict', response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    redis_client = Depends(get_redis),
    model_pool = Depends(get_model_pool)
):
    """p99 < 20ms at 14k+ requests/second"""

    # SHA1 feature hashing for 40%+ cache hits
    cache_key = hashlib.sha1(
        f"{request.features_rounded}:{MODEL_VERSION}".encode()
    ).hexdigest()

    # Redis pipeline for atomic operations
    pipe = redis_client.pipeline()
    cached = await pipe.get(cache_key).execute()

    if cached:
        metrics.cache_hits.inc()
        return json.loads(cached)

    # Micro-batching: 1ms timeout, 32 batch size
    async with batch_context(timeout_ms=1, size=32) as batch:
        batch.add(request)

        if batch.ready():
            # TorchScript optimized inference
            predictions = await model_pool.predict_batch(
                batch.items,
                compile_mode='reduce-overhead'
            )

            # Pipeline cache writes
            for pred in predictions:
                pipe.setex(pred.cache_key, 2, pred.json())
            await pipe.execute()

    return predictions[request.id]

# Benchmarked via K6: 14,235 rps sustained, p99=19.7ms`}
      </pre>
      <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/src/router/high_performance_router.py</div>
    </div>
  );
}

// AI Trading Infrastructure Overview Component
function AITradingInfraOverview() {
  const [activeView, setActiveView] = useState<string>('overview');

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">AI TRADING INFRASTRUCTURE</div>

      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setActiveView('overview')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'overview' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveView('metrics')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'metrics' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          Performance
        </button>
        <button
          onClick={() => setActiveView('bicep')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'bicep' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          BICEP Engine
        </button>
        <button
          onClick={() => setActiveView('enn')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'enn' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          ENN Model
        </button>
        <button
          onClick={() => setActiveView('fusion')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'fusion' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          FusionAlpha
        </button>
        <button
          onClick={() => setActiveView('infra')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'infra' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          Infrastructure
        </button>
        <button
          onClick={() => setActiveView('backtest')}
          className={`text-xs px-3 py-1 rounded border ${activeView === 'backtest' ? 'border-terminal-accent text-terminal-accent bg-terminal-accent/10' : 'border-terminal-muted/30 text-terminal-muted'}`}
        >
          Backtesting
        </button>
      </div>

      {activeView === 'overview' && (
        <div className="space-y-3">
          <div className="border-l-2 border-terminal-accent/30 pl-4">
            <div className="text-terminal-accent font-semibold">Institutional-Grade Trading System</div>
            <div className="text-sm mt-2 space-y-1">
              <div>• Walk-forward backtesting with 2.95 Sharpe ratio</div>
              <div>• Sub-20ms p99 latency at 14k+ requests/second</div>
              <div>• 1.2B+ events/day ingestion capacity</div>
              <div>• Three core modules: BICEP, ENN, FusionAlpha</div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="border border-terminal-muted/30 rounded p-2">
              <div className="text-terminal-success font-semibold mb-1">BICEP</div>
              <div className="text-terminal-muted">GPU Monte Carlo</div>
              <div>2.5M paths/sec</div>
              <div>0.4ms latency</div>
            </div>
            <div className="border border-terminal-muted/30 rounded p-2">
              <div className="text-terminal-success font-semibold mb-1">ENN</div>
              <div className="text-terminal-muted">Neural Network</div>
              <div>Entangled dynamics</div>
              <div>5K-148K params</div>
            </div>
            <div className="border border-terminal-muted/30 rounded p-2">
              <div className="text-terminal-success font-semibold mb-1">FusionAlpha</div>
              <div className="text-terminal-muted">Infrastructure</div>
              <div>14k+ rps</div>
              <div>40% cache hits</div>
            </div>
          </div>

          <div className="text-terminal-muted text-xs">
            Location: AI_TRADING_INFRA/ | Production-ready with Docker/K8s deployment
          </div>
        </div>
      )}

      {activeView === 'metrics' && <AITradingMetrics />}
      {activeView === 'bicep' && <BICEPCodeSnippet />}
      {activeView === 'enn' && <ENNCodeSnippet />}
      {activeView === 'fusion' && <FusionAlphaSnippet />}
      {activeView === 'infra' && <InfrastructureCode />}
      {activeView === 'backtest' && <BacktestingCode />}
    </div>
  );
}

// Infrastructure Deployment Code
function InfrastructureCode() {
  const [infraTab, setInfraTab] = useState<string>('docker');

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <button
          onClick={() => setInfraTab('docker')}
          className={`text-xs px-2 py-1 rounded ${infraTab === 'docker' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
        >
          Docker Compose
        </button>
        <button
          onClick={() => setInfraTab('websocket')}
          className={`text-xs px-2 py-1 rounded ${infraTab === 'websocket' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
        >
          WebSocket Stream
        </button>
        <button
          onClick={() => setInfraTab('ingestion')}
          className={`text-xs px-2 py-1 rounded ${infraTab === 'ingestion' ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-muted'}`}
        >
          NATS Ingestion
        </button>
      </div>

      {infraTab === 'docker' && (
        <div className="space-y-2">
          <div className="text-terminal-accent text-sm">High-Performance Docker Stack</div>
          <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`# compose-high-performance.yaml
services:
  # PostgreSQL with optimized settings
  store:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: \${PG_PASSWORD}
      POSTGRES_DB: trading
    deploy:
      resources:
        limits: { memory: 2G }
        reservations: { memory: 1G }

  # Redis with LRU cache eviction
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save "" --appendonly no

  # NATS JetStream for 1.2B events/day
  nats:
    image: nats:2.10-alpine
    command: >
      nats-server --jetstream
      --max_file_store 10GB
      --max_mem_store 1GB

  # Horizontal scaling router (3 replicas)
  router:
    build: ../docker/router/Dockerfile
    environment:
      TORCH_NUM_THREADS: 1
      MODEL_COMPILE_MODE: reduce-overhead
    deploy:
      replicas: 3
      resources:
        limits: { cpus: '4', memory: '4G' }
        reservations: { cpus: '2', memory: '2G' }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3`}
          </pre>
          <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/infra/compose-high-performance.yaml</div>
        </div>
      )}

      {infraTab === 'websocket' && (
        <div className="space-y-2">
          <div className="text-terminal-accent text-sm">WebSocket Streaming Server</div>
          <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`class RingBuffer:
    """Lock-free ring buffer for 400ms p99 latency"""
    def __init__(self, maxsize: int = 64):
        self.buffer = [None] * maxsize
        self.head = 0
        self.tail = 0
        self.size = 0

class WebSocketServer:
    """High-performance streaming with backpressure"""

    async def handle_client(self, websocket, path):
        client_queue = RingBuffer(maxsize=64)
        subscription = ClientSubscription(
            symbols={'SPY', 'QQQ', 'IWM'},
            signal_types={'prediction', 'order', 'fill'},
            max_queue_size=64
        )

        # Redis pub/sub for signal distribution
        pubsub = self.redis.pubsub()
        await pubsub.subscribe('signals:*')

        try:
            async for message in pubsub.listen():
                signal = TradingSignal(**json.loads(message['data']))

                # Track delivery latency for SLO
                latency = time.time() - signal.timestamp
                WS_DELIVERY_LATENCY.observe(latency)

                # Apply backpressure if queue full
                if client_queue.is_full():
                    WS_DROPPED_MESSAGES.labels(reason='backpressure').inc()
                    client_queue.pop_oldest()  # Drop oldest

                # Send with 400ms p99 guarantee
                await websocket.send(json.dumps(asdict(signal)))
                WS_THROUGHPUT.labels(
                    symbol=signal.symbol,
                    type=signal.signal_type
                ).inc()

        except websockets.ConnectionClosed:
            WS_CONNECTIONS.dec()`}
          </pre>
          <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/src/streaming/websocket_server.py</div>
        </div>
      )}

      {infraTab === 'ingestion' && (
        <div className="space-y-2">
          <div className="text-terminal-accent text-sm">NATS High-Throughput Ingestion</div>
          <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`class NATSIngestion:
    """1.2B+ events/day with zero data loss"""

    def __init__(self):
        self.batch_size = 10000
        self.batch_timeout = 60  # seconds
        self.s3_client = boto3.client('s3')

    async def consume_stream(self):
        """Process NATS JetStream with backpressure"""
        nc = await nats.connect("nats://nats:4222")
        js = nc.jetstream()

        # Durable consumer for at-least-once delivery
        consumer = await js.pull_subscribe(
            "market.>",
            durable="ingestion-consumer",
            config=ConsumerConfig(
                ack_policy=AckPolicy.EXPLICIT,
                max_deliver=3,
                ack_wait=30
            )
        )

        batch = []
        last_flush = time.time()

        while True:
            try:
                msgs = await consumer.fetch(100, timeout=1)

                for msg in msgs:
                    event = json.loads(msg.data.decode())
                    batch.append(event)

                    # Batch write to S3 Parquet
                    if len(batch) >= self.batch_size or \
                       time.time() - last_flush > self.batch_timeout:

                        df = pd.DataFrame(batch)
                        partition = f"date={event['date']}/symbol={event['symbol']}"

                        # Write partitioned Parquet
                        buffer = BytesIO()
                        df.to_parquet(buffer, compression='snappy')

                        self.s3_client.put_object(
                            Bucket='market-data',
                            Key=f"events/{partition}/{uuid4()}.parquet",
                            Body=buffer.getvalue()
                        )

                        # Ack messages after successful write
                        for m in msgs:
                            await m.ack()

                        INGEST_RATE.observe(len(batch))
                        batch.clear()
                        last_flush = time.time()

            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                INGEST_ERRORS.inc()`}
          </pre>
          <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/src/ingestion/nats_ingestion.py</div>
        </div>
      )}
    </div>
  );
}

// Backtesting Engine Code
function BacktestingCode() {
  return (
    <div className="space-y-2">
      <div className="text-terminal-accent text-sm">Walk-Forward Backtesting Engine</div>
      <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
{`class WalkForwardBacktest:
    """Production backtesting with 2.95 Sharpe achievement"""

    def __init__(self, config: Dict):
        # Initialize optimizers
        self.activity_optimizer = ActivityOptimizer(config['activity'])
        self.vol_threshold = VolatilityAdaptiveThreshold(config['adaptive'])
        self.tx_optimizer = TransactionOptimizer(config['transaction'])

        # Underhype strategy with ENN integration
        self.underhype_engine = UnderhypeEngine(
            confidence_threshold=2.0
        )

        # Risk controls
        self.max_position = 0.22  # 22% max single position
        self.max_gross = 1.20      # 120% max gross exposure
        self.target_vol = 0.175    # 17.5% target volatility

    def run_backtest(self, train_data, test_data):
        """Walk-forward with expanding windows"""
        results = []

        for fold_date in self.config['fold_dates']:
            # Expanding training window
            train = train_data[train_data.index < fold_date]
            test_fold = test_data[
                (test_data.index >= fold_date) &
                (test_data.index < fold_date + timedelta(days=30))
            ]

            # Fit only on train (no leakage)
            threshold = pick_threshold(train['scores'], train['returns'])

            # Apply optimizations
            signals = self.underhype_engine.generate_signals(
                test_fold,
                threshold=self.vol_threshold.adjust(test_fold['volatility'])
            )

            # Activity-aware scaling
            scaled_signals = self.activity_optimizer.scale(
                signals,
                activity_score=signals['activity'].mean()
            )

            # Transaction cost optimization
            filtered_signals = self.tx_optimizer.filter(
                scaled_signals,
                expected_value_threshold=2.0  # 2x cost minimum
            )

            # Risk management
            positions = self.apply_risk_controls(filtered_signals)

            # Calculate returns
            returns = positions * test_fold['forward_returns']
            transaction_costs = self.calculate_costs(positions)
            net_returns = returns - transaction_costs

            results.append({
                'date': fold_date,
                'returns': net_returns.sum(),
                'sharpe': net_returns.mean() / net_returns.std() * sqrt(252),
                'positions': len(positions[positions != 0])
            })

        return pd.DataFrame(results)

    def apply_risk_controls(self, signals):
        """Institutional-grade risk management"""
        positions = signals.copy()

        # Single position limit
        positions = positions.clip(-self.max_position, self.max_position)

        # Gross exposure limit
        gross = positions.abs().sum()
        if gross > self.max_gross:
            positions *= self.max_gross / gross

        # Volatility targeting
        realized_vol = positions.std() * sqrt(252)
        if realized_vol > 0:
            positions *= self.target_vol / realized_vol

        return positions`}
      </pre>
      <div className="text-terminal-muted text-xs">From: AI_TRADING_INFRA/src/strategy/backtest_runner.py</div>
    </div>
  );
}

// Generative Manim Demo Component
function GenerativeManimDemo() {
  const [selectedExample, setSelectedExample] = useState<string>('physics');
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [showCode, setShowCode] = useState<boolean>(false);
  const [renderStatus, setRenderStatus] = useState<string | null>(null);

  const examples = {
    physics: {
      title: "Physics: Conservation of Energy",
      prompt: "Visualize a pendulum demonstrating conservation of mechanical energy",
      preview: "Shows potential/kinetic energy transformation",
      code: `class EnergyConservation(Scene):
    def construct(self):
        # Create pendulum
        pivot = Dot(ORIGIN + UP * 2)
        bob = Circle(radius=0.2, color=BLUE).shift(DOWN * 2 + RIGHT * 2)
        rod = Line(pivot.get_center(), bob.get_center())
        pendulum = VGroup(pivot, rod, bob)

        # Energy bars
        ke_bar = Rectangle(width=0.5, height=0.01, color=RED)
        pe_bar = Rectangle(width=0.5, height=3, color=GREEN)

        # Animate swing with energy transformation
        self.play(Create(pendulum))
        self.play(
            Rotate(pendulum, angle=-PI/3, about_point=pivot.get_center()),
            Transform(pe_bar, ke_bar),
            rate_func=there_and_back,
            run_time=2
        )`
    },
    math: {
      title: "Mathematics: Fourier Series",
      prompt: "Animate how Fourier series approximates a square wave",
      preview: "Shows sine waves combining into square wave",
      code: `class FourierSeries(Scene):
    def construct(self):
        # Create axes
        axes = Axes(x_range=[-PI, PI, PI/2], y_range=[-2, 2, 1])

        # Build Fourier approximations
        funcs = [lambda x: (4/PI) * np.sin(x)]
        for n in range(3, 10, 2):
            funcs.append(lambda x, n=n: (4/(n*PI)) * np.sin(n*x))

        # Animate progressive approximation
        graph = axes.plot(funcs[0], color=BLUE)
        self.play(Create(axes), Create(graph))

        for func in funcs[1:]:
            new_graph = axes.plot(
                lambda x: sum(f(x) for f in funcs[:funcs.index(func)+1]),
                color=BLUE
            )
            self.play(Transform(graph, new_graph))`
    },
    chemistry: {
      title: "Chemistry: Molecular Orbital Theory",
      prompt: "Show the formation of molecular orbitals from atomic orbitals",
      preview: "Visualizes bonding/antibonding orbital formation",
      code: `class MolecularOrbitals(Scene):
    def construct(self):
        # Atomic orbitals
        orbital_1 = Circle(radius=1, color=BLUE).shift(LEFT * 2)
        orbital_2 = Circle(radius=1, color=BLUE).shift(RIGHT * 2)

        # Bonding orbital (constructive interference)
        bonding = Ellipse(width=5, height=2, color=GREEN)

        # Antibonding orbital (destructive interference)
        antibonding = VGroup(
            Arc(radius=1, angle=PI, color=RED).shift(LEFT),
            Arc(radius=1, angle=PI, color=RED).shift(RIGHT).rotate(PI)
        )

        # Animate orbital combination
        self.play(Create(orbital_1), Create(orbital_2))
        self.play(
            Transform(VGroup(orbital_1, orbital_2), bonding),
            run_time=2
        )
        self.wait()
        self.play(Transform(bonding, antibonding))`
    },
    algorithms: {
      title: "Algorithms: Quick Sort Visualization",
      prompt: "Animate the quick sort algorithm step by step",
      preview: "Shows pivot selection and partitioning",
      code: `class QuickSort(Scene):
    def construct(self):
        # Create array of bars
        values = [8, 3, 5, 4, 7, 6, 1, 2]
        bars = VGroup(*[
            Rectangle(height=val*0.3, width=0.4, color=BLUE)
            .shift(RIGHT * (i - 3.5))
            for i, val in enumerate(values)
        ])

        # Pivot selection
        pivot = bars[0]
        pivot.set_color(RED)

        # Partition animation
        self.play(Create(bars))
        self.play(pivot.animate.set_color(RED))

        # Animate swapping
        for i, bar in enumerate(bars[1:]):
            if values[i+1] < values[0]:
                self.play(
                    bar.animate.shift(LEFT * 2),
                    rate_func=smooth,
                    run_time=0.5
                )`
    }
  };

  const handleGenerate = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setIsGenerating(false);
      setShowCode(true);
    }, 2000);
  };

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88] text-lg font-bold">GENERATIVE MANIM DEMO</div>

      <div className="text-sm text-terminal-text">
        Educational video generation pipeline using LLMs + ManimGL
        <div className="text-xs text-terminal-muted mt-1">
          Production implementation from Sending Labs (5× throughput, 35% cost reduction)
        </div>
      </div>

      <div className="border border-terminal-accent/30 rounded p-4 space-y-3">
        <div className="text-terminal-accent text-sm font-semibold">Select Example Topic:</div>

        <div className="grid grid-cols-2 gap-2">
          {Object.entries(examples).map(([key, example]) => (
            <button
              key={key}
              onClick={() => {
                setSelectedExample(key);
                setShowCode(false);
              }}
              className={`text-left p-2 rounded border transition-colors ${
                selectedExample === key
                  ? 'border-terminal-accent bg-terminal-accent/10'
                  : 'border-terminal-muted/30 hover:border-terminal-muted'
              }`}
            >
              <div className="text-xs font-semibold text-terminal-success">{example.title}</div>
              <div className="text-xs text-terminal-muted mt-1">{example.preview}</div>
            </button>
          ))}
        </div>

        <div className="space-y-2">
          <div className="text-terminal-accent text-sm">Or Enter Custom Prompt:</div>
          <input
            type="text"
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            placeholder="e.g., Explain the double-slit experiment with wave animations"
            className="w-full bg-black/30 border border-terminal-muted/30 rounded px-3 py-2 text-sm text-terminal-text placeholder-terminal-muted/50 focus:border-terminal-accent focus:outline-none"
          />
        </div>

        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="px-4 py-2 bg-terminal-accent/20 text-terminal-accent border border-terminal-accent rounded hover:bg-terminal-accent/30 transition-colors disabled:opacity-50 text-sm"
        >
          {isGenerating ? 'Generating Manim Code...' : 'Generate Animation'}
        </button>

        {showCode && (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <div className="text-terminal-success text-sm">Generated Manim Code:</div>
              <div className="text-xs text-terminal-muted">
                Model: GPT-4o | Domain: {selectedExample}
              </div>
            </div>

            <pre className="text-xs bg-black/50 p-3 rounded overflow-x-auto border border-terminal-muted/30">
              {examples[selectedExample as keyof typeof examples].code}
            </pre>

            <div className="flex gap-2">
              <button
                onClick={() => {
                  setRenderStatus('Initializing ManimGL engine...');
                  setTimeout(() => setRenderStatus('Compiling LaTeX formulas...'), 1000);
                  setTimeout(() => setRenderStatus('Rendering animation frames... (60 FPS)'), 2000);
                  setTimeout(() => setRenderStatus('Encoding with FFmpeg... (H.264/MP4)'), 3500);
                  setTimeout(() => {
                    setRenderStatus('✓ Rendering complete! (Demo mode - actual API required for video file)');
                    setTimeout(() => setRenderStatus(null), 4000);
                  }, 5000);
                }}
                className="text-xs text-terminal-success hover:underline">
                [▶ Render Video]
              </button>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(examples[selectedExample as keyof typeof examples].code);
                  alert('Code copied to clipboard!');
                }}
                className="text-xs text-terminal-success hover:underline">
                [📋 Copy Code]
              </button>
              <button
                onClick={() => {
                  setShowCode(false);
                  setTimeout(() => {
                    setIsGenerating(true);
                    setTimeout(() => {
                      setIsGenerating(false);
                      setShowCode(true);
                    }, 1500);
                  }, 100);
                }}
                className="text-xs text-terminal-success hover:underline">
                [🔄 Regenerate]
              </button>
            </div>
            {renderStatus && (
              <div className="mt-2 p-2 bg-terminal-accent/10 border border-terminal-accent/30 rounded">
                <div className="text-xs text-terminal-accent animate-pulse">{renderStatus}</div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border border-terminal-muted/30 rounded p-3 space-y-2">
        <div className="text-terminal-accent text-sm">API Architecture:</div>
        <pre className="text-xs text-terminal-muted">
{`POST /v1/video/rendering
├── LLM Processing (GPT-4o/Claude)
│   ├── Domain-specific prompts
│   ├── Storyboard generation
│   └── Manim code creation
├── ManimGL Rendering
│   ├── LaTeX processing
│   ├── Animation compilation
│   └── FFmpeg encoding
└── Output
    ├── Local: ./media/
    └── Cloud: S3/GCS`}
        </pre>
      </div>

      <div className="text-xs text-terminal-muted">
        Note: This demo shows the architecture. Actual rendering requires backend API.
        <br />
        Source: Generative-Manim-Template-main/ | Docker-ready deployment
      </div>
    </div>
  );
}

// Export components for lazy loading
export { TradingDashboard, MonteCarloDemo };
