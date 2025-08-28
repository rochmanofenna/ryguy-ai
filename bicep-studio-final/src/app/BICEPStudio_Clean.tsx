import React, { useEffect, useMemo, useRef, useState } from 'react';
import { generate, bands, GRID_SIZE } from "../engine/mapFactory";
import { 
  Play, RotateCcw, Brain, Loader2, CheckCircle, BarChart3
} from "lucide-react";
import type { MapFamily, MapParams, SamplerParams, Winner } from "../engine/types";

const CELL = 13;
const SERVER_URL = "http://localhost:8000";

export default function BICEPStudio() {
  // Core state
  const [family, setFamily] = useState<MapFamily>('corridors');
  const [mapSeed, setMapSeed] = useState(654321);
  const [grid, setGrid] = useState(() => generate('corridors', { width: 4 }, 654321));
  const sg = useMemo(() => bands(), []);

  const [params, setParams] = useState<SamplerParams>({
    K: 5000, T: 300, mu: 1.2, sigma: 0.7, rho: 0,
    antithetic: false, wind: { angle: 0, strength: 0 }, batch: 2000
  });

  // Canvas refs
  const bgRef = useRef<HTMLCanvasElement>(null);
  const obsRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<HTMLCanvasElement>(null);
  const pathRef = useRef<HTMLCanvasElement>(null);

  // Simulation state
  const [winners, setWinners] = useState<Winner[]>([]);
  const [active, setActive] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  // Training state
  const [currentPolicyId, setCurrentPolicyId] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<string>('');
  const [evalResult, setEvalResult] = useState<string>('');
  const [serverOnline, setServerOnline] = useState(false);
  const [epochs, setEpochs] = useState(100);
  const [baselineResult, setBaselineResult] = useState<string>('');

  // Check server on startup
  useEffect(() => {
    checkServer();
    const interval = setInterval(checkServer, 10000);
    return () => clearInterval(interval);
  }, []);

  async function checkServer() {
    try {
      const response = await fetch(`${SERVER_URL}/health`);
      setServerOnline(response.ok);
    } catch {
      setServerOnline(false);
    }
  }

  // Dynamic canvas sizing
  const [canvasSize, setCanvasSize] = useState({ width: GRID_SIZE * CELL, height: GRID_SIZE * CELL });
  
  useEffect(() => {
    const updateCanvasSize = () => {
      const container = bgRef.current?.parentElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const size = Math.min(rect.width, rect.height);
        setCanvasSize({ width: size, height: size });
      }
    };
    
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);

  // Draw grid background
  useEffect(() => {
    const canvas = bgRef.current!;
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;
    const ctx = canvas.getContext('2d')!;
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const cellWidth = canvas.width / GRID_SIZE;
    const cellHeight = canvas.height / GRID_SIZE;
    
    ctx.strokeStyle = 'rgba(255,255,255,.03)';
    for (let i = 0; i <= GRID_SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellWidth, 0);
      ctx.lineTo(i * cellWidth, canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cellHeight);
      ctx.lineTo(canvas.width, i * cellHeight);
      ctx.stroke();
    }
  }, [canvasSize]);

  // Draw obstacles and zones
  const drawObstacles = () => {
    const canvas = obsRef.current!;
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cellWidth = canvas.width / GRID_SIZE;
    const cellHeight = canvas.height / GRID_SIZE;

    // Start zone (green)
    ctx.fillStyle = 'rgba(34,197,94,.15)';
    sg.start.forEach(([x, y]) => {
      ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
    });

    // Goal zone (blue)
    ctx.fillStyle = 'rgba(59,130,246,.15)';
    sg.goal.forEach(([x, y]) => {
      ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
    });

    // Obstacles (red)
    ctx.fillStyle = 'rgba(220,38,38,.8)';
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        if (grid[x][y]) {
          ctx.fillRect(x * cellWidth + 1, y * cellHeight + 1, cellWidth - 2, cellHeight - 2);
        }
      }
    }
  };

  useEffect(drawObstacles, [grid, canvasSize]);

  // Worker setup
  const workerRef = useRef<Worker | null>(null);
  
  useEffect(() => {
    workerRef.current = new Worker(
      new URL('../engine/bicep.worker.ts', import.meta.url), 
      { type: 'module' }
    );
    return () => workerRef.current?.terminate();
  }, []);

  function runBICEP() {
    if (isRunning) return;
    
    setWinners([]);
    setIsRunning(true);
    
    const worker = workerRef.current!;
    worker.onmessage = (e: any) => {
      const msg = e.data;
      if (msg.type === 'tick') {
        const { winners, active } = msg.payload;
        setWinners(winners);
        setActive(active);
        drawSim(msg.payload);
      } else if (msg.type === 'done') {
        setIsRunning(false);
      }
    };
    
    worker.postMessage({
      type: 'run',
      grid, start: sg.start, goal: sg.goal,
      seed: Date.now(), params
    });
  }

  function drawSim(payload: { heat: Float32Array; winners: Winner[] }) {
    // Heat map
    const simCanvas = simRef.current!;
    const simCtx = simCanvas.getContext('2d')!;
    simCanvas.width = canvasSize.width;
    simCanvas.height = canvasSize.height;

    const cellWidth = simCanvas.width / GRID_SIZE;
    const cellHeight = simCanvas.height / GRID_SIZE;

    const { heat } = payload;
    const maxHeat = Math.max(...Array.from(heat));
    if (maxHeat > 0) {
      for (let x = 0; x < GRID_SIZE; x++) {
        for (let y = 0; y < GRID_SIZE; y++) {
          const intensity = heat[y * GRID_SIZE + x] / maxHeat;
          if (intensity > 0) {
            simCtx.fillStyle = `rgba(16,185,129,${intensity * 0.2})`;
            simCtx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
          }
        }
      }
    }

    // Best path
    const pathCanvas = pathRef.current!;
    const pathCtx = pathCanvas.getContext('2d')!;
    pathCanvas.width = canvasSize.width;
    pathCanvas.height = canvasSize.height;
    pathCtx.clearRect(0, 0, pathCanvas.width, pathCanvas.height);

    if (winners.length > 0) {
      const bestPath = winners[0].path;
      pathCtx.strokeStyle = '#10b981';
      pathCtx.lineWidth = Math.max(2, cellWidth / 6);
      pathCtx.lineCap = 'round';
      pathCtx.beginPath();

      bestPath.forEach(([gx, gy], i) => {
        const x = gx * cellWidth + cellWidth / 2;
        const y = gy * cellHeight + cellHeight / 2;
        if (i === 0) pathCtx.moveTo(x, y);
        else pathCtx.lineTo(x, y);
      });
      pathCtx.stroke();
    }
  }

  function newMap() {
    setMapSeed(Math.floor(Math.random() * 1e6));
    const newGrid = generate(family, { width: 4 }, mapSeed);
    setGrid(newGrid);
  }

  async function trainENN() {
    if (!serverOnline || isTraining) return;
    
    setIsTraining(true);
    setTrainingProgress('Training on demonstrations...');
    setEvalResult('');
    
    try {
      const response = await fetch(`${SERVER_URL}/enn/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          family,
          map_seed: mapSeed,
          bicep_params: { K: params.K, T: params.T, mu: params.mu, sigma: params.sigma },
          num_demos: 100,
          bc_epochs: epochs,
          rl_episodes: 0
        })
      });

      const result = await response.json();
      setCurrentPolicyId(result.policy_id);
      setTrainingProgress(`Training complete! Policy: ${result.policy_id.substring(0,6)}...`);
      
      setTimeout(() => {
        setTrainingProgress('');
        setIsTraining(false);
      }, 2000);

    } catch (error) {
      setTrainingProgress('Training failed');
      setTimeout(() => {
        setTrainingProgress('');
        setIsTraining(false);
      }, 2000);
    }
  }

  async function evaluate() {
    if (!currentPolicyId || !serverOnline) return;

    try {
      const response = await fetch(`${SERVER_URL}/enn/eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          policy_id: currentPolicyId,
          test_seeds: [111, 222, 333, 444, 555],
          episodes_per_seed: 10
        })
      });

      const result = await response.json();
      const successRate = (result.overall_success_rate * 100).toFixed(0);
      const avgSteps = result.mean_steps ? result.mean_steps.toFixed(1) : 'N/A';
      setEvalResult(`${successRate}% success | ${avgSteps} avg steps`);

    } catch (error) {
      setEvalResult('Eval failed');
    }
  }

  async function evaluateBaseline() {
    if (!serverOnline) return;

    try {
      // Create a simple baseline policy (random or greedy)
      const response = await fetch(`${SERVER_URL}/enn/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          family: 'corridors',
          map_seed: 12345,
          num_demos: 10,  // Very few demos
          bc_epochs: 1,   // Minimal training
          rl_episodes: 0
        })
      });

      const trainResult = await response.json();
      
      // Evaluate the baseline
      const evalResponse = await fetch(`${SERVER_URL}/enn/eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          policy_id: trainResult.policy_id,
          test_seeds: [111, 222, 333, 444, 555],
          episodes_per_seed: 10
        })
      });

      const evalResult = await evalResponse.json();
      const successRate = (evalResult.overall_success_rate * 100).toFixed(0);
      setBaselineResult(`Baseline: ${successRate}% success`);

    } catch (error) {
      setBaselineResult('Baseline eval failed');
    }
  }

  return (
    <div className="p-4 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6 text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-400 to-blue-400 bg-clip-text text-transparent mb-2">
          BICEP · ENN Demo
        </h1>
        <p className="text-zinc-400 text-sm">
          Path sampling → Neural network training → Navigation
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6">
        {/* Canvas */}
        <div className="relative">
          <div 
            className="relative rounded-lg overflow-hidden border border-white/10 bg-black/20 w-full"
            style={{ aspectRatio: '4/3', minHeight: '400px' }}
          >
            <canvas ref={bgRef} className="absolute inset-0" />
            <canvas ref={obsRef} className="absolute inset-0" />
            <canvas ref={simRef} className="absolute inset-0" />
            <canvas ref={pathRef} className="absolute inset-0" />
            
            {/* Status overlay */}
            <div className="absolute top-3 left-3 text-xs bg-black/60 px-3 py-1 rounded">
              {active > 0 && <span className="text-emerald-400">{active.toLocaleString()} particles</span>}
              {winners.length > 0 && (
                <span className="ml-3">
                  Best: <span className="text-blue-400">{winners[0].cost.toFixed(1)}</span> cost
                </span>
              )}
            </div>

            {/* Policy status */}
            {currentPolicyId && (
              <div className="absolute top-3 right-3 text-xs bg-blue-600/20 border border-blue-500/30 px-3 py-1 rounded">
                Policy: {currentPolicyId.substring(0,8)}
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="space-y-4">
          {/* Map */}
          <div className="bg-zinc-900/50 border border-white/10 rounded-lg p-4">
            <h3 className="text-emerald-400 font-medium mb-3">Map</h3>
            <div className="grid grid-cols-2 gap-2 mb-3">
              <button 
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  family === 'corridors' ? 'bg-emerald-600' : 'bg-zinc-700 hover:bg-zinc-600'
                }`}
                onClick={() => setFamily('corridors')}
              >
                Corridors
              </button>
              <button 
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  family === 'mazes' ? 'bg-emerald-600' : 'bg-zinc-700 hover:bg-zinc-600'
                }`}
                onClick={() => setFamily('mazes')}
              >
                Mazes
              </button>
              <button 
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  family === 'rooms_doors' ? 'bg-emerald-600' : 'bg-zinc-700 hover:bg-zinc-600'
                }`}
                onClick={() => setFamily('rooms_doors')}
              >
                Rooms
              </button>
              <button 
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  family === 'random_obstacles' ? 'bg-emerald-600' : 'bg-zinc-700 hover:bg-zinc-600'
                }`}
                onClick={() => { setFamily('random_obstacles'); }}
              >
                Random
              </button>
            </div>
            <button 
              className="w-full bg-zinc-700 hover:bg-zinc-600 px-3 py-2 rounded text-sm font-medium transition-colors flex items-center justify-center"
              onClick={newMap}
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              New Map
            </button>
          </div>

          {/* BICEP */}
          <div className="bg-zinc-900/50 border border-white/10 rounded-lg p-4">
            <h3 className="text-emerald-400 font-medium mb-3">Path Sampling</h3>
            <div className="mb-3">
              <div className="flex justify-between text-sm mb-1">
                <span>Particles (K)</span>
                <span className="text-emerald-400">{params.K.toLocaleString()}</span>
              </div>
              <input 
                type="range" 
                min="1000" 
                max="10000" 
                step="1000" 
                value={params.K} 
                onChange={e => setParams(p => ({ ...p, K: parseInt(e.target.value) }))}
                className="w-full"
              />
            </div>
            <button 
              className={`w-full px-3 py-2 rounded text-sm font-medium transition-colors flex items-center justify-center ${
                isRunning 
                  ? 'bg-red-600 hover:bg-red-500' 
                  : 'bg-emerald-600 hover:bg-emerald-500'
              }`}
              onClick={runBICEP}
            >
              <Play className="w-4 h-4 mr-2" />
              {isRunning ? 'Running...' : 'Run BICEP'}
            </button>
          </div>

          {/* Training */}
          <div className="bg-zinc-900/50 border border-white/10 rounded-lg p-4">
            <h3 className="text-blue-400 font-medium mb-3">Neural Network</h3>
            
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm">Server</span>
              <div className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-green-400' : 'bg-red-400'}`} />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-sm mb-1">
                <span>Training Epochs</span>
                <span className="text-blue-400">{epochs}</span>
              </div>
              <input 
                type="range" 
                min="10" 
                max="500" 
                step="10" 
                value={epochs} 
                onChange={e => setEpochs(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <button 
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-3 py-2 rounded text-sm font-medium transition-colors flex items-center justify-center mb-2"
              onClick={trainENN}
              disabled={!serverOnline || isTraining}
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4 mr-2" />
                  Train ENN
                </>
              )}
            </button>

            <div className="grid grid-cols-2 gap-2">
              <button 
                className="bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 px-3 py-2 rounded text-sm font-medium transition-colors flex items-center justify-center"
                onClick={evaluate}
                disabled={!currentPolicyId}
              >
                <BarChart3 className="w-4 h-4 mr-1" />
                Evaluate
              </button>
              <button 
                className="bg-orange-600 hover:bg-orange-500 disabled:opacity-50 px-3 py-2 rounded text-sm font-medium transition-colors"
                onClick={evaluateBaseline}
                disabled={!serverOnline}
              >
                Baseline
              </button>
            </div>

            {trainingProgress && (
              <div className="mt-3 p-2 bg-blue-800/20 border border-blue-600/30 rounded text-xs flex items-center">
                {isTraining && <Loader2 className="w-3 h-3 mr-2 animate-spin" />}
                {!isTraining && <CheckCircle className="w-3 h-3 mr-2 text-green-400" />}
                {trainingProgress}
              </div>
            )}

            {evalResult && (
              <div className="mt-2 p-2 bg-green-800/20 border border-green-600/30 rounded text-xs">
                <div className="font-medium">Trained Policy</div>
                {evalResult}
              </div>
            )}

            {baselineResult && (
              <div className="mt-2 p-2 bg-orange-800/20 border border-orange-600/30 rounded text-xs">
                {baselineResult}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}