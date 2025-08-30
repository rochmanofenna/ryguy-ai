import type { Grid, Point, SamplerParams, Winner, WorkerMsg, WorkerEvt } from "./types";

const GRID = 40;

let grid: Grid = [];
let start: Point[] = [];
let goal: Point[] = [];
let params: SamplerParams;
let horizon = 300;
let rng: number;
let particles: any[] = [];
let winners: Winner[] = [];
let step = 0;
let K = 0;

self.onmessage = (e: MessageEvent<WorkerMsg>) => {
  const msg = e.data;
  
  if (msg.type === "run") {
    grid = msg.grid;
    start = msg.start;
    goal = msg.goal;
    params = msg.params;
    horizon = params.T;
    rng = msg.seed >>> 0;
    K = params.K;
    
    initParticles();
    step = 0;
    winners = [];
    
    // Start simulation loop
    tick();
  } else if (msg.type === "step") {
    tick();
  } else if (msg.type === "reset") {
    particles = [];
    winners = [];
    step = 0;
  } else if (msg.type === "stop") {
    particles = [];
  }
};

function initParticles() {
  particles.length = 0;
  
  for (let i = 0; i < K; i++) {
    const startPos = start[Math.floor(rand() * start.length)];
    particles.push({
      id: i,
      x: startPos[0] + 0.5,
      y: startPos[1] + 0.5,
      alive: true,
      reached: false,
      cost: 0,
      path: [[startPos[0], startPos[1]]],
      t: 0,
      lastNoise: null
    });
  }
}

function tick() {
  const startTime = performance.now();
  const { sigma, mu, rho, antithetic, wind } = params;
  const dt = 1.0 / horizon;
  const sqrtDt = Math.sqrt(dt);
  
  let active = 0;
  let reached = 0;
  const heat = new Float32Array(GRID * GRID);
  
  for (let idx = 0; idx < particles.length; idx++) {
    const p = particles[idx];
    if (!p.alive || p.reached) continue;
    
    active++;
    
    // Find nearest goal
    let nearestGoal = goal[0];
    let minDist = Infinity;
    
    for (const g of goal) {
      const dx = g[0] - p.x;
      const dy = g[1] - p.y;
      const dist = dx * dx + dy * dy;
      if (dist < minDist) {
        minDist = dist;
        nearestGoal = g;
      }
    }
    
    // Calculate drift towards goal
    const remaining = Math.max(dt, 1 - p.t);
    const pullX = mu * (nearestGoal[0] + 0.5 - p.x) / remaining * dt;
    const pullY = mu * (nearestGoal[1] + 0.5 - p.y) / remaining * dt;
    
    // Generate noise
    let noiseX: number, noiseY: number;
    
    if (antithetic && idx < K / 2) {
      // Use antithetic sampling for variance reduction
      const twinId = idx + Math.floor(K / 2);
      const twin = particles[twinId];
      
      if (twin && twin.lastNoise) {
        noiseX = -twin.lastNoise.x;
        noiseY = -twin.lastNoise.y;
      } else {
        const z1 = gauss();
        const z2 = gauss();
        noiseX = z1 * sigma * sqrtDt;
        noiseY = (rho * z1 + Math.sqrt(1 - rho * rho) * z2) * sigma * sqrtDt;
        p.lastNoise = { x: noiseX, y: noiseY };
      }
    } else {
      const z1 = gauss();
      const z2 = gauss();
      noiseX = z1 * sigma * sqrtDt;
      noiseY = (rho * z1 + Math.sqrt(1 - rho * rho) * z2) * sigma * sqrtDt;
      p.lastNoise = { x: noiseX, y: noiseY };
    }
    
    // Apply wind
    const windAngleRad = (wind.angle * Math.PI) / 180;
    const windX = Math.cos(windAngleRad) * wind.strength * dt;
    const windY = Math.sin(windAngleRad) * wind.strength * dt;
    
    // Update position
    p.x += pullX + noiseX + windX;
    p.y += pullY + noiseY + windY;
    p.t += dt;
    
    // Clamp to grid bounds
    p.x = Math.max(0, Math.min(GRID - 1, p.x));
    p.y = Math.max(0, Math.min(GRID - 1, p.y));
    
    const ix = Math.floor(p.x);
    const iy = Math.floor(p.y);
    
    // Update heat map
    heat[iy * GRID + ix] += 1;
    
    // Check collision
    if (grid[ix][iy]) {
      p.alive = false;
      continue;
    }
    
    // Add to path
    p.path.push([ix, iy]);
    
    // Calculate proximity cost
    let proximityCost = 0;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const nx = ix + dx;
        const ny = iy + dy;
        if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && grid[nx][ny]) {
          proximityCost += 0.1;
        }
      }
    }
    
    p.cost += 1 + proximityCost;
    
    // Check if reached goal
    for (const g of goal) {
      if (g[0] === ix && g[1] === iy) {
        p.reached = true;
        reached++;
        recordWinner(p);
        break;
      }
    }
  }
  
  step++;
  const frameMs = performance.now() - startTime;
  const throughput = active > 0 ? Math.round(active / (frameMs / 1000)) : 0;
  
  const payload = {
    heat,
    winners: winners.slice(0, 50), // Send top 50
    active,
    reached,
    step,
    frameMs,
    throughput
  };
  
  (self as any).postMessage({
    type: "tick",
    payload
  } as WorkerEvt);
  
  if (active === 0 || step >= horizon) {
    (self as any).postMessage({ type: "done" } as WorkerEvt);
  }
}

function recordWinner(p: any) {
  // Calculate risk based on path clearance
  let totalRisk = 0;
  let minClearance = Infinity;
  
  for (const [x, y] of p.path) {
    let localClearance = Infinity;
    
    for (let dx = -3; dx <= 3; dx++) {
      for (let dy = -3; dy <= 3; dy++) {
        if (dx === 0 && dy === 0) continue;
        
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && grid[nx][ny]) {
          const clearance = Math.sqrt(dx * dx + dy * dy);
          localClearance = Math.min(localClearance, clearance);
        }
      }
    }
    
    minClearance = Math.min(minClearance, localClearance);
    
    // Risk penalty: exponential in low clearance
    if (localClearance < 2.5) {
      totalRisk += Math.exp(2.5 - localClearance) * 0.1;
    }
  }
  
  const riskScore = totalRisk + (minClearance < 1.5 ? 2.0 : 0);
  
  winners.push({
    path: [...p.path],
    cost: p.cost,
    risk: riskScore,
    clearance: minClearance,
    id: p.id
  });
  
  // Sort by cost (or risk if using CVaR)
  winners.sort((a, b) => a.cost - b.cost);
  
  // Keep only top 100 for memory efficiency
  if (winners.length > 100) {
    winners.length = 100;
  }
}

function rand(): number {
  rng = Math.imul(rng, 1664525) + 1013904223 | 0;
  return (rng >>> 0) / 4294967296;
}

function gauss(): number {
  const u = rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}