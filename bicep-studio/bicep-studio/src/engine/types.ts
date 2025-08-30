export type Grid = boolean[][]; // [x][y]
export type Point = [number, number];
export type Winner = { 
  path: Point[]; 
  cost: number; 
  risk: number; 
  clearance: number; 
  id: number 
};

export type SamplerParams = {
  K: number;
  T: number;
  mu: number;
  sigma: number;
  rho: number;
  antithetic: boolean;
  wind: { angle: number; strength: number };
  batch: number;
};

export type MapFamily = "corridors" | "rooms_doors" | "mazes" | "random_obstacles";
export type MapParams = { width?: number; density?: number };

export type WorkerMsg =
  | { type: "init"; gridSize: number; cellSize: number }
  | { type: "run"; grid: Grid; start: Point[]; goal: Point[]; seed: number; params: SamplerParams }
  | { type: "step" }
  | { type: "reset" }
  | { type: "stop" };

export type WorkerEvt =
  | { 
      type: "tick"; 
      payload: { 
        heat: Float32Array; 
        winners: Winner[]; 
        active: number; 
        reached: number; 
        step: number; 
        frameMs: number; 
        throughput: number;
      } 
    }
  | { type: "done" };