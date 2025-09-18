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
              <div><span className="text-terminal-success">about</span> → Professional summary</div>
              <div><span className="text-terminal-success">story</span> → CS:GO origin story</div>
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
            <TypewriterText text={`NYU CS/Math + Philosophy

TECH STACK:
• Languages: Python, C/C++ (CUDA), Rust, SQL, JavaScript, TypeScript
• ML/AI: PyTorch, JAX, TensorFlow, scikit-learn, Transformers
• Systems: CUDA/Triton, Docker, Kubernetes, Linux, Git
• Infrastructure: AWS (EC2/S3/Lambda), PostgreSQL, Redis, Kafka

CURRENT PROJECTS:
• Trading research stack - Sub-20ms p99 latency, 1B+ events/day
• EEG neural pipeline - 129-channel processing on 8×V100 cluster
• GPU Monte Carlo engine - 10× speedup over NumPy baseline
• Custom neural network (ENN) - 98% accuracy gesture recognition

RECENT EXPERIENCE:
• Systems Engineer @ Stealth Trading (2024-Present)
• ML Engineer @ Sending Labs (Jun-Aug 2024)
• ML Engineer @ Video Tutor AI (Apr-Jun 2024)
• Software Engineering Intern @ Olo (2022-2023)

[Type 'story' for the CS:GO origin story]
[Type 'projects' for detailed project info]
[Type 'experience' for full timeline]`} />
          ),
          type: 'output',
          timestamp
        }]);
        break;

      case 'portfolio':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <TradingDashboard onInteraction={setInteractiveMode} />,
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
          content: <MonteCarloDemo onInteraction={setInteractiveMode} />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'skills':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <SkillsProof onInteraction={setAwaitingInput} />,
          type: 'output',
          timestamp
        }]);
        break;

      case 'projects':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: <ProjectsList onInteraction={setAwaitingInput} />,
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

      case 'story':
        setLines(prev => [...prev, {
          id: `output-${Date.now()}`,
          content: (
            <TypewriterText text={`I was getting destroyed in CS:GO. Not just my aim - my frames. 30 FPS on dust2,
stuttering during firefights, basically unplayable. My laptop had 12 cores but
played like a potato.

Then I opened Activity Monitor mid-game. One core at 100%. Eleven cores doing nothing.

The fix wasn't upgrading - it was understanding. I learned that Source engine's main
thread was bottlenecked on single-core performance. But I could force other processes
off that core, disable CPU throttling, tune memory timings. Went from 30 to 60+ FPS
on the same "trash" laptop.

The performance boost basically made me go pro - climbed from Silver 2 (bottom 5th
percentile) to Silver Elite Master (bottom 10th percentile). Okay, still terrible,
but 2× the frames meant I could finally blame my aim instead of my hardware.

That moment changed everything. I realized most "slow" computers aren't slow -
they're just badly utilized.

This obsession with squeezing performance out of hardware led me deeper. CS:GO taught
me about CPU scheduling and cache locality. Then I applied it to ML training - why was
NumPy so slow? Because it wasn't compiled with OpenBLAS. Set OMP_NUM_THREADS=12,
suddenly my models trained 10× faster.

The pattern was always the same: the hardware could do more, I just had to unlock it.

Now I build trading systems that process billions of events at sub-20ms latency.
I write CUDA kernels that outperform NumPy by 10×. But it all started with trying
to hit headshots at more than 30 FPS.

The path from gaming to quant trading isn't as weird as it sounds. Both care about
every microsecond. Both punish inefficiency. Both reward understanding your hardware
at the metal level.

That's my edge - I learned to optimize on hardware most people threw away. Now I
apply that same obsession to systems where microseconds mean millions.`} />
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

// Experience Timeline Component
function ExperienceTimeline() {
  return (
    <div className="space-y-2">
      <div className="text-[#00FF88]">=== EXPERIENCE TIMELINE ===</div>
      <div>[2024-NOW] Stealth Buy-Side Research Stack.......[ACTIVE]</div>
      <div className="ml-4 text-xs text-terminal-muted">
        Architected end-to-end research/execution stack
        Sub-20ms p99 latency | Real capital deployment
      </div>
      <div>[2024-JUN] Sending Labs...........................[3 MOS]</div>
      <div className="ml-4 text-xs text-terminal-muted">
        ManimGL pipeline | Modal/Fly.io | 5× throughput, 35% cost↓
      </div>
      <div>[2024-APR] Video Tutor AI.........................[3 MOS]</div>
      <div className="ml-4 text-xs text-terminal-muted">
        GPT-4o + TTS + ManimGL | High-concurrency pipeline
      </div>
      <div>[2022-2023] Olo...................................[INTERN]</div>
      <div className="ml-4 text-xs text-terminal-muted">
        Infrastructure automation | Terraform IaC | Observability integration
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

// Skills Proof Component
function SkillsProof({ onInteraction }: { onInteraction: (mode: string | null) => void }) {
  useEffect(() => {
    onInteraction('skill-select');
  }, [onInteraction]);

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88]">=== TECHNICAL STACK ===</div>
      <div className="space-y-3">
        <div>
          <div className="text-terminal-accent">→ Mathematics</div>
          <div className="ml-4 text-sm">
            • Linear Algebra, Probability, Optimization, Numerical Methods<br/>
            • Stochastic Differential Equations for path simulation<br/>
            • Cryptography (Number Theory), Discrete Math, Combinatorics
          </div>
        </div>
        <div>
          <div className="text-terminal-accent">→ Systems & Infrastructure</div>
          <div className="ml-4 text-sm">
            • CUDA/Triton kernels, custom GPU implementations<br/>
            • Docker, Kubernetes, Ray, AWS (EC2, S3, Lambda)<br/>
            • NYU Greene HPC: 8×V100 cluster optimization
          </div>
        </div>
        <div>
          <div className="text-terminal-accent">→ Languages & ML</div>
          <div className="ml-4 text-sm">
            • Python, C/C++ (CUDA/low-latency), Rust, SQL, JavaScript<br/>
            • PyTorch, JAX, TensorFlow, scikit-learn, Optuna<br/>
            • GNNs, Transformers, Neural ODEs
          </div>
        </div>
      </div>
      <div className="text-terminal-muted text-xs mt-4 animate-pulse">
        Type skill name (e.g., 'python', 'cuda') for code examples, or 'validate' for GitHub proof
      </div>
    </div>
  );
}

// Projects List Component
function ProjectsList({ onInteraction }: { onInteraction: (mode: string | null) => void }) {
  useEffect(() => {
    onInteraction('project-select');
  }, [onInteraction]);

  return (
    <div className="space-y-4">
      <div className="text-[#00FF88]">=== PROJECT PORTFOLIO ===</div>
      <div className="space-y-4">
        <div className="text-terminal-accent">RESEARCH & INFRASTRUCTURE</div>

        <div className="border-l-2 border-terminal-muted/30 pl-4 space-y-3">
          <div>
            <div className="text-terminal-text">1. Stealth Buy-Side Research Stack</div>
            <div className="text-xs text-terminal-muted ml-3">
              • Python, Rust, PostgreSQL, Redis, Kubernetes
              • Sub-20ms p99 latency on 1B+ events/day
            </div>
          </div>

          <div>
            <div className="text-terminal-text">2. EEG 2025: Contradiction-Aware Neural Decoding</div>
            <div className="text-xs text-terminal-muted ml-3">
              • PyTorch, JAX, 8×V100 cluster, MNE-Python
              • Published at NYU Greene HPC, 129-channel processing
            </div>
          </div>

          <div>
            <div className="text-terminal-text">3. GPU Monte Carlo Engine (BICEP)</div>
            <div className="text-xs text-terminal-muted ml-3">
              • CUDA, Triton, NumPy, ChaCha20/AES-CTR PRGs
              • 10× speedup over NumPy baseline
            </div>
          </div>

          <div>
            <div className="text-terminal-text">4. Custom Neural Network (ENN)</div>
            <div className="text-xs text-terminal-muted ml-3">
              • C++17, Eigen3, OpenMP, AVX2 vectorization
              • Gesture recognition with real-time inference
            </div>
          </div>
        </div>

        <div className="text-terminal-accent mt-4">SYSTEMS & APPLICATIONS</div>

        <div className="border-l-2 border-terminal-muted/30 pl-4 space-y-3">
          <div>
            <div className="text-terminal-text">5. ManimGL Pipeline @ Sending Labs</div>
            <div className="text-xs text-terminal-muted ml-3">
              • Python, ManimGL, Modal, Fly.io, Docker
              • 5× throughput improvement, 35% cost reduction
            </div>
          </div>

          <div>
            <div className="text-terminal-text">6. Video Tutor AI</div>
            <div className="text-xs text-terminal-muted ml-3">
              • GPT-4, Whisper, ElevenLabs TTS, Next.js, TypeScript
              • High-concurrency educational content generation
            </div>
          </div>

          <div>
            <div className="text-terminal-text">7. Infrastructure Automation @ Olo</div>
            <div className="text-xs text-terminal-muted ml-3">
              • Terraform, C#, SQL, Cloudflare, Datadog
              • IaC migration and observability integration
            </div>
          </div>

          <div>
            <div className="text-terminal-text">8. NYU Tandon Made Challenge Winner</div>
            <div className="text-xs text-terminal-muted ml-3">
              • React, TypeScript, Node.js, MongoDB
              • Competition winner among 50+ teams
            </div>
          </div>
        </div>
      </div>
      <div className="text-terminal-muted text-xs mt-4 animate-pulse">
        Enter number (1-8) or search by tech (e.g., 'cuda', 'python', 'ml')
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
// Export components for lazy loading
export { TradingDashboard, MonteCarloDemo };
