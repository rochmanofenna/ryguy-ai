import React, { useEffect, useMemo, useRef, useState } from 'react';
import { act, loadPolicyFromFile } from "../engine/policy";
import { generate, bands, GRID_SIZE } from "../engine/mapFactory";
import { stateToUrl, urlToState } from "../lib/seedUrl";
import { 
  Play, RotateCcw, Upload, BarChart3, Share2, 
  Image as ImageIcon, Pause, Brain, Target, Zap,
  Settings, Info, Activity
} from "lucide-react";
import type { Ensemble } from "../engine/policy";
import type { MapFamily, MapParams, SamplerParams, Winner } from "../engine/types";

const CELL = 13;
const ENHANCED_SERVER_URL = 'http://localhost:8001';

interface NavigationResult {
  success: boolean;
  path: number[][];
  length: number;
  confidence?: number;
  ensemble_votes?: Array<{expert: number, action: string, confidence: number}>;
  features_analysis?: {
    goal_distance: number;
    obstacle_density: number;
    path_clarity: number;
    progress: number;
  };
  time_taken: number;
}

export default function EnhancedBICEPStudio() {
  // Existing state
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
  const navRef = useRef<HTMLCanvasElement>(null);

  // Enhanced navigation state
  const [navigationMethod, setNavigationMethod] = useState<'astar' | 'bicep' | 'enn_bicep'>('enn_bicep');
  const [showConfidence, setShowConfidence] = useState(true);
  const [startPos, setStartPos] = useState<[number, number]>([2, 2]);
  const [goalPos, setGoalPos] = useState<[number, number]>([45, 45]);
  const [navigationResult, setNavigationResult] = useState<NavigationResult | null>(null);
  const [isNavigating, setIsNavigating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [ennTrained, setEnnTrained] = useState(false);
  const [clickMode, setClickMode] = useState<'start' | 'goal' | 'obstacle' | 'navigate'>('navigate');

  // Existing simulation state
  const [winners, setWinners] = useState<Winner[]>([]);
  const [active, setActive] = useState(0);
  const [reached, setReached] = useState(0);
  const [step, setStep] = useState(0);
  const [throughput, setThroughput] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  // Policy state
  const [ens, setEns] = useState<Ensemble | null>(null);
  const [policyStats, setPolicyStats] = useState<string>('');

  // Check server status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${ENHANCED_SERVER_URL}/api/status`);
        if (response.ok) {
          const status = await response.json();
          setEnnTrained(status.enn_trained);
        }
      } catch (error) {
        console.log('Enhanced server not available');
      }
    };
    checkStatus();
  }, []);

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

    // Draw goal zone (red)
    ctx.fillStyle = 'rgba(239,68,68,.2)';
    sg.goal.forEach(([x, y]) => {
      ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    });

    // Draw obstacles (white)
    ctx.fillStyle = 'rgba(255,255,255,.9)';
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (grid[y][x] === 1) {
          ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
        }
      }
    }

    // Draw navigation start/goal
    ctx.fillStyle = 'rgba(0,255,0,0.8)';
    ctx.fillRect(startPos[0] * CELL, startPos[1] * CELL, CELL, CELL);
    
    ctx.fillStyle = 'rgba(255,0,0,0.8)';
    ctx.fillRect(goalPos[0] * CELL, goalPos[1] * CELL, CELL, CELL);
  };

  // Draw navigation path
  const drawNavigationPath = () => {
    if (!navigationResult?.path) return;
    
    const canvas = navRef.current!;
    canvas.width = GRID_SIZE * CELL;
    canvas.height = GRID_SIZE * CELL;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw path
    if (navigationResult.path.length > 1) {
      ctx.strokeStyle = navigationMethod === 'enn_bicep' ? '#3b82f6' : 
                       navigationMethod === 'astar' ? '#10b981' : '#f59e0b';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      const [startX, startY] = navigationResult.path[0];
      ctx.moveTo(startX * CELL + CELL/2, startY * CELL + CELL/2);
      
      for (let i = 1; i < navigationResult.path.length; i++) {
        const [x, y] = navigationResult.path[i];
        ctx.lineTo(x * CELL + CELL/2, y * CELL + CELL/2);
      }
      ctx.stroke();

      // Draw path points
      ctx.fillStyle = ctx.strokeStyle;
      navigationResult.path.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * CELL + CELL/2, y * CELL + CELL/2, 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw confidence visualization for ENN+BICEP
    if (navigationMethod === 'enn_bicep' && showConfidence && navigationResult.confidence) {
      const confidence = navigationResult.confidence;
      const alpha = confidence * 0.3;
      ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
      
      navigationResult.path.forEach(([x, y]) => {
        ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
      });
    }
  };

  // Handle canvas clicks
  const handleCanvasClick = (event: React.MouseEvent) => {
    const canvas = bgRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / CELL);
    const y = Math.floor((event.clientY - rect.top) / CELL);

    if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) return;

    if (clickMode === 'start') {
      setStartPos([x, y]);
    } else if (clickMode === 'goal') {
      setGoalPos([x, y]);
    } else if (clickMode === 'obstacle') {
      // Toggle obstacle
      const newGrid = grid.map(row => [...row]);
      newGrid[y][x] = newGrid[y][x] === 1 ? 0 : 1;
      setGrid(newGrid);
    } else if (clickMode === 'navigate') {
      // Quick navigation to clicked point
      setGoalPos([x, y]);
      setTimeout(() => navigate(), 100);
    }
  };

  // Train ENN+BICEP
  const trainENN = async () => {
    setIsTraining(true);
    try {
      const response = await fetch(`${ENHANCED_SERVER_URL}/api/train-enn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grid_size: GRID_SIZE,
          obstacle_density: 0.2,
          num_demos: 300,
          epochs: 75
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          setEnnTrained(true);
          alert('ENN+BICEP trained successfully!');
        }
      }
    } catch (error) {
      console.error('Training failed:', error);
      alert('Training failed. Make sure enhanced server is running on port 8001.');
    } finally {
      setIsTraining(false);
    }
  };

  // Navigate with selected method
  const navigate = async () => {
    setIsNavigating(true);
    try {
      const response = await fetch(`${ENHANCED_SERVER_URL}/api/navigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grid: grid,
          start: [startPos[0], startPos[1]],
          goal: [goalPos[0], goalPos[1]],
          method: navigationMethod,
          show_confidence: showConfidence
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setNavigationResult(result);
      } else {
        const error = await response.json();
        alert(`Navigation failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Navigation failed:', error);
      alert('Navigation failed. Make sure enhanced server is running on port 8001.');
    } finally {
      setIsNavigating(false);
    }
  };

  // Update drawing when state changes
  useEffect(() => {
    drawObstacles();
    drawNavigationPath();
  }, [grid, startPos, goalPos, navigationResult, navigationMethod, showConfidence]);

  const generateMap = () => {
    const newGrid = generate(family, mparams, mapSeed);
    setGrid(newGrid);
    setNavigationResult(null);
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-blue-400 flex items-center gap-2">
            <Brain className="w-8 h-8" />
            Enhanced BICEP Studio
            <span className="text-lg text-gray-400">with ENN+BICEP Navigation</span>
          </h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas Area */}
          <div className="lg:col-span-2">
            <div className="bg-gray-900 p-4 rounded-lg">
              <div className="relative border border-gray-700">
                <canvas 
                  ref={bgRef} 
                  className="absolute inset-0 cursor-crosshair"
                  onClick={handleCanvasClick}
                />
                <canvas ref={obsRef} className="absolute inset-0 pointer-events-none" />
                <canvas ref={simRef} className="absolute inset-0 pointer-events-none" />
                <canvas ref={pathRef} className="absolute inset-0 pointer-events-none" />
                <canvas ref={navRef} className="absolute inset-0 pointer-events-none" />
              </div>
              
              {/* Navigation Results */}
              {navigationResult && (
                <div className="mt-4 p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center gap-4 mb-2">
                    <span className={`px-2 py-1 rounded text-sm ${
                      navigationResult.success ? 'bg-green-600' : 'bg-red-600'
                    }`}>
                      {navigationResult.success ? '✓ Success' : '✗ Failed'}
                    </span>
                    <span>Length: {navigationResult.length}</span>
                    <span>Time: {navigationResult.time_taken.toFixed(3)}s</span>
                    {navigationResult.confidence && (
                      <span className="text-blue-400">
                        Confidence: {(navigationResult.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  
                  {/* ENN+BICEP specific info */}
                  {navigationMethod === 'enn_bicep' && navigationResult.ensemble_votes && (
                    <div className="mt-2 text-xs">
                      <div className="mb-1 text-gray-300">Ensemble Votes:</div>
                      <div className="flex gap-2">
                        {navigationResult.ensemble_votes.map(vote => (
                          <span key={vote.expert} className="bg-blue-900 px-2 py-1 rounded">
                            Expert {vote.expert}: {vote.action} ({(vote.confidence * 100).toFixed(0)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {navigationResult.features_analysis && (
                    <div className="mt-2 text-xs">
                      <div className="mb-1 text-gray-300">Spatial Analysis:</div>
                      <div className="grid grid-cols-4 gap-2">
                        <span>Distance: {navigationResult.features_analysis.goal_distance.toFixed(2)}</span>
                        <span>Obstacles: {navigationResult.features_analysis.obstacle_density.toFixed(2)}</span>
                        <span>Clarity: {navigationResult.features_analysis.path_clarity.toFixed(2)}</span>
                        <span>Progress: {navigationResult.features_analysis.progress.toFixed(2)}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Enhanced Control Panel */}
          <div className="space-y-6">
            {/* Navigation Controls */}
            <div className="bg-gray-900 p-4 rounded-lg">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Target className="w-5 h-5 text-green-400" />
                Navigation Control
              </h3>
              
              {/* Method Selection */}
              <div className="mb-4">
                <label className="block text-sm text-gray-300 mb-2">Method</label>
                <select 
                  value={navigationMethod} 
                  onChange={(e) => setNavigationMethod(e.target.value as any)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                >
                  <option value="enn_bicep">🧠 ENN+BICEP (Learned)</option>
                  <option value="astar">⭐ A* (Optimal)</option>
                  <option value="bicep">🎲 BICEP (Stochastic)</option>
                </select>
              </div>

              {/* Click Mode */}
              <div className="mb-4">
                <label className="block text-sm text-gray-300 mb-2">Click Mode</label>
                <div className="grid grid-cols-2 gap-2">
                  {['start', 'goal', 'obstacle', 'navigate'].map(mode => (
                    <button
                      key={mode}
                      onClick={() => setClickMode(mode as any)}
                      className={`px-3 py-2 rounded text-sm capitalize ${
                        clickMode === mode ? 'bg-blue-600' : 'bg-gray-700'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {/* Position Display */}
              <div className="mb-4 text-sm">
                <div>Start: ({startPos[0]}, {startPos[1]})</div>
                <div>Goal: ({goalPos[0]}, {goalPos[1]})</div>
              </div>

              {/* ENN Training */}
              {!ennTrained && (
                <button
                  onClick={trainENN}
                  disabled={isTraining}
                  className="w-full mb-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded flex items-center justify-center gap-2"
                >
                  <Brain className="w-4 h-4" />
                  {isTraining ? 'Training ENN...' : 'Train ENN+BICEP'}
                </button>
              )}

              {ennTrained && (
                <div className="mb-4 p-2 bg-green-900 rounded text-center text-sm">
                  ✅ ENN+BICEP Ready
                </div>
              )}

              {/* Navigate Button */}
              <button
                onClick={navigate}
                disabled={isNavigating || (navigationMethod === 'enn_bicep' && !ennTrained)}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded flex items-center justify-center gap-2"
              >
                <Zap className="w-4 h-4" />
                {isNavigating ? 'Navigating...' : 'Navigate'}
              </button>

              {/* Confidence Toggle */}
              {navigationMethod === 'enn_bicep' && (
                <div className="mt-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showConfidence}
                      onChange={(e) => setShowConfidence(e.target.checked)}
                    />
                    <span className="text-sm">Show Confidence</span>
                  </label>
                </div>
              )}
            </div>

            {/* Map Controls */}
            <div className="bg-gray-900 p-4 rounded-lg">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Settings className="w-5 h-5 text-yellow-400" />
                Map Generation
              </h3>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-sm text-gray-300 mb-1">Family</label>
                  <select 
                    value={family} 
                    onChange={(e) => setFamily(e.target.value as MapFamily)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                  >
                    <option value="corridors">Corridors</option>
                    <option value="rooms">Rooms & Doors</option>
                    <option value="maze">Maze</option>
                    <option value="random">Random</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-300 mb-1">Seed</label>
                  <input
                    type="number"
                    value={mapSeed}
                    onChange={(e) => setMapSeed(parseInt(e.target.value) || 0)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                  />
                </div>

                <button
                  onClick={generateMap}
                  className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded flex items-center justify-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Generate Map
                </button>
              </div>
            </div>

            {/* Method Comparison */}
            <div className="bg-gray-900 p-4 rounded-lg">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Info className="w-5 h-5 text-blue-400" />
                Method Comparison
              </h3>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-green-400">A* (Optimal)</span>
                  <span>Fast, Shortest Path</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-yellow-400">BICEP (Stochastic)</span>
                  <span>Exploratory, Robust</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-400">ENN+BICEP</span>
                  <span>Learned, Confident</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}