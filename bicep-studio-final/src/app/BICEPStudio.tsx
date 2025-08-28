import React, { useEffect, useMemo, useRef, useState } from 'react';
import { act, loadPolicyFromFile } from "../engine/policy";
import { generate, bands, GRID_SIZE } from "../engine/mapFactory";
import { stateToUrl, urlToState } from "../lib/seedUrl";
import { 
  Play, RotateCcw, Upload, BarChart3, Share2, 
  Image as ImageIcon, Pause
} from "lucide-react";
import type { Ensemble } from "../engine/policy";
import type { MapFamily, MapParams, SamplerParams, Winner } from "../engine/types";

const CELL = 13;

export default function BICEPStudio() {
  // State management
  const [family, setFamily] = useState<MapFamily>('corridors');
  const [mparams, setMParams] = useState<MapParams>({ width: 4, density: 0.2 });
  const [seed, setSeed] = useState(123456);
  const [mapSeed, setMapSeed] = useState(654321);
  const [grid, setGrid] = useState(() => generate('corridors', { width: 4 }, 654321));
  const sg = useMemo(() => bands(), []);

  const [params, setParams] = useState<SamplerParams>({
    K: 5000,
    T: 300,
    mu: 1.2,
    sigma: 0.7,
    rho: 0,
    antithetic: false,
    wind: { angle: 0, strength: 0 },
    batch: 2000
  });

  // Canvas refs
  const bgRef = useRef<HTMLCanvasElement>(null);
  const obsRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<HTMLCanvasElement>(null);
  const pathRef = useRef<HTMLCanvasElement>(null);

  // Simulation state
  const [winners, setWinners] = useState<Winner[]>([]);
  const [active, setActive] = useState(0);
  const [reached, setReached] = useState(0);
  const [step, setStep] = useState(0);
  const [throughput, setThroughput] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  // Policy state
  const [ens, setEns] = useState<Ensemble | null>(null);
  const [policyStats, setPolicyStats] = useState<string>('');

  // Initialize from URL params
  useEffect(() => {
    const urlState = urlToState();
    if (urlState.seed) setSeed(urlState.seed);
    if (urlState.mapSeed) setMapSeed(urlState.mapSeed);
    if (urlState.family) setFamily(urlState.family);
    if (urlState.params) {
      setParams(prev => ({ ...prev, ...urlState.params }));
    }
  }, []);

  // Draw static background grid
  useEffect(() => {
    const canvas = bgRef.current!;
    canvas.width = GRID_SIZE * CELL;
    canvas.height = GRID_SIZE * CELL;
    const ctx = canvas.getContext('2d')!;
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = 'rgba(255,255,255,.04)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= GRID_SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(i * CELL, 0);
      ctx.lineTo(i * CELL, canvas.height);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, i * CELL);
      ctx.lineTo(canvas.width, i * CELL);
      ctx.stroke();
    }
  }, []);

  // Draw obstacles and zones
  const drawObstacles = () => {
    const canvas = obsRef.current!;
    canvas.width = GRID_SIZE * CELL;
    canvas.height = GRID_SIZE * CELL;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw start zone (green)
    ctx.fillStyle = 'rgba(34,197,94,.2)';
    sg.start.forEach(([x, y]) => {
      ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    });

    // Draw goal zone (blue)
    ctx.fillStyle = 'rgba(59,130,246,.2)';
    sg.goal.forEach(([x, y]) => {
      ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    });

    // Draw obstacles (red)
    ctx.fillStyle = 'rgba(220,38,38,.7)';
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        if (grid[x][y]) {
          ctx.fillRect(x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2);
        }
      }
    }
  };

  useEffect(drawObstacles, [grid]);

  // Worker setup
  const workerRef = useRef<Worker | null>(null);
  
  useEffect(() => {
    workerRef.current = new Worker(
      new URL('../engine/bicep.worker.ts', import.meta.url), 
      { type: 'module' }
    );
    
    return () => workerRef.current?.terminate();
  }, []);

  function run() {
    if (isRunning) return;
    
    setWinners([]);
    setIsRunning(true);
    
    const worker = workerRef.current!;
    worker.onmessage = (e: any) => {
      const msg = e.data;
      if (msg.type === 'tick') {
        const { winners, active, reached, step, throughput } = msg.payload;
        setWinners(winners);
        setActive(active);
        setReached(reached);
        setStep(step);
        setThroughput(throughput);
        drawSim(msg.payload);
      } else if (msg.type === 'done') {
        setIsRunning(false);
      }
    };
    
    worker.postMessage({
      type: 'run',
      grid,
      start: sg.start,
      goal: sg.goal,
      seed,
      params
    });
  }

  function stop() {
    const worker = workerRef.current!;
    worker.postMessage({ type: 'stop' });
    setIsRunning(false);
  }

  function drawSim(payload: { heat: Float32Array; winners: Winner[]; active: number; reached: number; step: number; throughput: number }) {
    // Clear simulation canvas with slight fade
    const simCanvas = simRef.current!;
    const simCtx = simCanvas.getContext('2d')!;
    simCanvas.width = GRID_SIZE * CELL;
    simCanvas.height = GRID_SIZE * CELL;
    simCtx.fillStyle = 'rgba(0,0,0,.06)';
    simCtx.fillRect(0, 0, simCanvas.width, simCanvas.height);

    // Draw heat map
    const { heat } = payload;
    const maxHeat = Math.max(...Array.from(heat));
    if (maxHeat > 0) {
      for (let x = 0; x < GRID_SIZE; x++) {
        for (let y = 0; y < GRID_SIZE; y++) {
          const intensity = heat[y * GRID_SIZE + x] / maxHeat;
          if (intensity > 0) {
            simCtx.fillStyle = `rgba(16,185,129,${intensity * 0.3})`;
            simCtx.fillRect(x * CELL, y * CELL, CELL, CELL);
          }
        }
      }
    }

    // Draw best path
    const pathCanvas = pathRef.current!;
    const pathCtx = pathCanvas.getContext('2d')!;
    pathCanvas.width = simCanvas.width;
    pathCanvas.height = simCanvas.height;
    pathCtx.clearRect(0, 0, pathCanvas.width, pathCanvas.height);

    if (winners.length > 0) {
      const bestPath = winners[0].path;
      pathCtx.shadowColor = 'rgba(16,185,129,.6)';
      pathCtx.shadowBlur = 12;
      pathCtx.strokeStyle = '#10b981';
      pathCtx.lineWidth = 3;
      pathCtx.lineCap = 'round';
      pathCtx.beginPath();

      bestPath.forEach(([gx, gy], i) => {
        const x = gx * CELL + CELL / 2;
        const y = gy * CELL + CELL / 2;
        if (i === 0) pathCtx.moveTo(x, y);
        else pathCtx.lineTo(x, y);
      });

      pathCtx.stroke();
      pathCtx.shadowBlur = 0;
    }
  }

  function regen() {
    const newGrid = generate(family, mparams, mapSeed);
    setGrid(newGrid);
    drawObstacles();
  }

  async function evaluatePolicy() {
    if (!ens) {
      alert('Load a policy first');
      return;
    }

    const episodes = 20;
    let successes = 0;
    let totalSteps = 0;

    for (let ep = 0; ep < episodes; ep++) {
      const startPos = sg.start[Math.floor(Math.random() * sg.start.length)];
      let x = startPos[0];
      let y = startPos[1];
      let steps = 0;
      let done = false;

      while (!done && steps < params.T) {
        const action = act(ens, x, y, grid);
        const [dx, dy] = [[1, 0], [0, 1], [-1, 0], [0, -1]][action];
        const nx = Math.max(0, Math.min(GRID_SIZE - 1, x + dx));
        const ny = Math.max(0, Math.min(GRID_SIZE - 1, y + dy));

        if (grid[nx][ny]) {
          done = true; // Hit wall -> fail
          break;
        }

        x = nx;
        y = ny;
        steps++;

        if (sg.goal.some(g => g[0] === x && g[1] === y)) {
          successes++;
          totalSteps += steps;
          done = true;
        }
      }
    }

    const successRate = (successes / episodes * 100).toFixed(1);
    const avgSteps = successes > 0 ? (totalSteps / successes).toFixed(1) : 'N/A';
    const stats = `Success: ${successRate}% | Avg steps: ${avgSteps}`;
    setPolicyStats(stats);
  }

  function copyShareLink() {
    const url = stateToUrl({
      seed,
      mapSeed,
      family,
      params: {
        K: params.K,
        T: params.T,
        mu: params.mu,
        sigma: params.sigma,
        rho: params.rho
      }
    });
    
    navigator.clipboard.writeText(url);
    alert('Link copied to clipboard!');
  }

  return (
    <div className="p-6 max-w-[1400px] mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-emerald-300 bg-clip-text text-transparent">
          BICEP · ENN Navigation Studio
        </h1>
        <p className="text-zinc-400 mt-2">
          Parallel path sampling + ensemble policy evaluation with deterministic replays
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
        {/* Main Canvas Area */}
        <div className="relative">
          <div 
            className="relative rounded-lg overflow-hidden border border-white/10 bg-black/20"
            style={{ height: GRID_SIZE * CELL }}
          >
            <canvas ref={bgRef} className="absolute inset-0" />
            <canvas ref={obsRef} className="absolute inset-0" />
            <canvas ref={simRef} className="absolute inset-0" />
            <canvas ref={pathRef} className="absolute inset-0" />
            
            {/* Overlay info */}
            <div className="absolute top-3 left-3 text-xs bg-emerald-500/10 border border-emerald-500/40 px-3 py-1 rounded-full">
              seed <span className="font-mono font-bold">{seed}</span> · map <span className="font-mono font-bold">{mapSeed}</span>
            </div>
            
            <div className="absolute top-3 right-3 text-xs bg-black/60 border border-white/10 px-3 py-1 rounded">
              active <span className="font-bold text-emerald-400">{active}</span> · 
              reached <span className="font-bold text-blue-400">{reached}</span> · 
              t <span className="font-bold">{step}/{params.T}</span>
              {throughput > 0 && <span className="ml-2 text-zinc-400">({throughput}/s)</span>}
            </div>

            {winners.length > 0 && (
              <div className="absolute bottom-3 left-3 text-xs bg-black/60 border border-white/10 px-3 py-1 rounded">
                Best path: <span className="font-bold text-emerald-400">{winners[0].cost.toFixed(2)}</span> cost · 
                <span className="font-bold text-orange-400">{winners[0].risk.toFixed(2)}</span> risk
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <aside className="space-y-4">
          {/* Map Section */}
          <Section title="Map Generation">
            <div className="grid grid-cols-2 gap-2">
              <Button 
                variant={family === 'corridors' && mparams.width === 4 ? 'primary' : 'secondary'}
                onClick={() => { setFamily('corridors'); setMParams(p => ({ ...p, width: 4 })); }}
              >
                Corridors W=4
              </Button>
              <Button 
                variant={family === 'corridors' && mparams.width === 2 ? 'primary' : 'secondary'}
                onClick={() => { setFamily('corridors'); setMParams(p => ({ ...p, width: 2 })); }}
              >
                Corridors W=2
              </Button>
              <Button 
                variant={family === 'rooms_doors' ? 'primary' : 'secondary'}
                onClick={() => setFamily('rooms_doors')}
              >
                Rooms & Doors
              </Button>
              <Button 
                variant={family === 'mazes' ? 'primary' : 'secondary'}
                onClick={() => setFamily('mazes')}
              >
                Mazes
              </Button>
              <Button 
                variant={family === 'random_obstacles' && mparams.density === 0.2 ? 'primary' : 'secondary'}
                onClick={() => { setFamily('random_obstacles'); setMParams(p => ({ ...p, density: 0.2 })); }}
              >
                Random 20%
              </Button>
              <Button 
                variant={family === 'random_obstacles' && mparams.density === 0.4 ? 'primary' : 'secondary'}
                onClick={() => { setFamily('random_obstacles'); setMParams(p => ({ ...p, density: 0.4 })); }}
              >
                Random 40%
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-3">
              <Button variant="secondary" onClick={() => {
                setMapSeed(Math.floor(Math.random() * 1e6));
                setTimeout(regen, 0);
              }}>
                <RotateCcw className="w-4 h-4 mr-1" />
                New Map
              </Button>
              <Button variant="secondary" onClick={regen}>
                Apply Changes
              </Button>
            </div>
          </Section>

          {/* Sampler Section */}
          <Section title="BICEP Sampler">
            <Slider 
              label="K (particles)" 
              value={params.K} 
              min={1000} 
              max={20000} 
              step={1000} 
              onChange={v => setParams(p => ({ ...p, K: v }))} 
            />
            <Slider 
              label="T (horizon)" 
              value={params.T} 
              min={100} 
              max={500} 
              step={50} 
              onChange={v => setParams(p => ({ ...p, T: v }))} 
            />
            <Slider 
              label="μ (drift)" 
              value={params.mu} 
              min={0.1} 
              max={3.0} 
              step={0.1} 
              onChange={v => setParams(p => ({ ...p, mu: v }))} 
            />
            <Slider 
              label="σ (noise)" 
              value={params.sigma} 
              min={0.1} 
              max={2.0} 
              step={0.1} 
              onChange={v => setParams(p => ({ ...p, sigma: v }))} 
            />
            <Slider 
              label="ρ (correlation)" 
              value={params.rho} 
              min={-0.9} 
              max={0.9} 
              step={0.1} 
              onChange={v => setParams(p => ({ ...p, rho: v }))} 
            />
            
            <div className="grid grid-cols-2 gap-2 mt-3">
              <Button 
                variant="primary" 
                onClick={isRunning ? stop : run}
                disabled={false}
              >
                {isRunning ? (
                  <>
                    <Pause className="w-4 h-4 mr-1" />
                    Stop
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-1" />
                    Run BICEP
                  </>
                )}
              </Button>
              <Button variant="secondary" onClick={() => {
                setSeed(Math.floor(Math.random() * 1e6));
              }}>
                New Seed
              </Button>
            </div>
          </Section>

          {/* Policy Section */}
          <Section title="ENN Policy">
            <div className="grid grid-cols-2 gap-2">
              <Button variant="secondary" onClick={async () => {
                const ensemble = await loadPolicyFromFile();
                if (ensemble) {
                  setEns(ensemble);
                  alert(`Policy loaded: ${ensemble.heads.length} heads, ${ensemble.hidden} hidden`);
                }
              }}>
                <Upload className="w-4 h-4 mr-1" />
                Load Policy
              </Button>
              <Button 
                variant="secondary" 
                onClick={evaluatePolicy}
                disabled={!ens}
              >
                <BarChart3 className="w-4 h-4 mr-1" />
                Evaluate
              </Button>
            </div>
            
            {policyStats && (
              <div className="mt-2 p-2 bg-zinc-800 border border-zinc-700 rounded text-xs">
                {policyStats}
              </div>
            )}
            
            {ens && (
              <div className="mt-2 text-xs text-zinc-400">
                Loaded: {ens.heads.length} heads, {ens.hidden} hidden units
              </div>
            )}
          </Section>

          {/* Export Section */}
          <Section title="Share & Export">
            <div className="grid grid-cols-2 gap-2">
              <Button variant="secondary" onClick={copyShareLink}>
                <Share2 className="w-4 h-4 mr-1" />
                Copy Link
              </Button>
              <Button variant="secondary" onClick={() => {
                exportPNG(bgRef.current!, obsRef.current!, simRef.current!, pathRef.current!);
              }}>
                <ImageIcon className="w-4 h-4 mr-1" />
                Export PNG
              </Button>
            </div>
          </Section>
        </aside>
      </div>
    </div>
  );
}

// Helper Components
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-zinc-900/50 border border-white/10 rounded-lg p-4">
      <div className="text-emerald-400 text-sm font-semibold mb-3">{title}</div>
      {children}
    </div>
  );
}

function Button({ 
  variant = 'primary', 
  children, 
  onClick, 
  disabled = false 
}: { 
  variant?: 'primary' | 'secondary'; 
  children: React.ReactNode; 
  onClick?: () => void; 
  disabled?: boolean;
}) {
  const baseClasses = "inline-flex items-center justify-center px-3 py-2 rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
  const variantClasses = {
    primary: "bg-emerald-600 hover:bg-emerald-500 disabled:hover:bg-emerald-600",
    secondary: "bg-zinc-700 hover:bg-zinc-600 disabled:hover:bg-zinc-700"
  };

  return (
    <button 
      className={`${baseClasses} ${variantClasses[variant]}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between text-sm mb-1">
        <span className="text-zinc-300">{label}</span>
        <span className="text-emerald-400 font-semibold font-mono">
          {typeof value === 'number' ? value.toFixed(step < 1 ? 1 : 0) : value}
        </span>
      </div>
      <input 
        type="range" 
        min={min} 
        max={max} 
        step={step} 
        value={value} 
        onChange={e => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

function exportPNG(...layers: HTMLCanvasElement[]) {
  const w = layers[0].width;
  const h = layers[0].height;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  
  for (const layer of layers) {
    ctx.drawImage(layer, 0, 0);
  }
  
  const link = document.createElement('a');
  link.download = `bicep-${Date.now()}.png`;
  link.href = canvas.toDataURL();
  link.click();
}