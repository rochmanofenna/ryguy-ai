import { mulberry32 } from "../lib/rng";
import type { Grid, MapFamily, MapParams, Point } from "./types";

const GRID = 40;

export function bands() {
  const start: Point[] = [];
  const goal: Point[] = [];
  
  // Start band on left side
  for (let y = 10; y < 30; y++) {
    for (let x = 0; x < 3; x++) {
      start.push([x, y]);
    }
  }
  
  // Goal band on right side
  for (let y = 10; y < 30; y++) {
    for (let x = GRID - 3; x < GRID; x++) {
      goal.push([x, y]);
    }
  }
  
  return { start, goal };
}

export function generate(family: MapFamily, params: MapParams, seed: number): Grid {
  const rng = mulberry32(seed);
  const grid = Array.from({ length: GRID }, () => Array(GRID).fill(false));
  
  if (family === "corridors") {
    const w = params.width ?? 4;
    const cy = GRID >> 1;
    const half = w >> 1;
    
    // Create walls above and below corridor
    for (let x = 3; x < GRID - 3; x++) {
      for (let y = 0; y < cy - half; y++) {
        grid[x][y] = true;
      }
      for (let y = cy + half + 1; y < GRID; y++) {
        grid[x][y] = true;
      }
    }
    
    // Add some random obstacles in the corridor
    for (let i = 0; i < Math.floor(rng() * w); i++) {
      const x = 5 + Math.floor(rng() * (GRID - 10));
      const y = cy - half + Math.floor(rng() * w);
      grid[x][y] = true;
    }
  } else if (family === "rooms_doors") {
    // Start with walls everywhere
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        grid[x][y] = true;
      }
    }
    
    // Carve out rooms
    const baseRooms = [
      [5, 5, 8, 8],
      [20, 8, 10, 6],
      [15, 25, 12, 8]
    ] as const;
    
    for (const [x0, y0, w, h] of baseRooms) {
      const x = x0 + ri(rng, -2, 2);
      const y = y0 + ri(rng, -2, 2);
      const W = w + ri(rng, -1, 1);
      const H = h + ri(rng, -1, 1);
      
      for (let i = x; i < x + W; i++) {
        for (let j = y; j < y + H; j++) {
          if (i >= 0 && i < GRID && j >= 0 && j < GRID) {
            grid[i][j] = false;
          }
        }
      }
    }
    
    // Create doors between rooms
    for (let x = 13; x < 20; x++) {
      if (rng() > 0.2) {
        grid[x][10] = false;
        grid[x][11] = false;
      }
    }
    
    for (let y = 14; y < 25; y++) {
      if (rng() > 0.2) {
        grid[22][y] = false;
        grid[23][y] = false;
      }
    }
  } else if (family === "mazes") {
    // Create vertical maze structure
    for (let i = 0; i < 12; i++) {
      const x = 5 + i * 3;
      for (let y = 0; y < GRID; y++) {
        if (rng() > 0.3) {
          grid[x][y] = true;
        }
      }
    }
    
    // Add horizontal walls
    for (let y = 8; y < GRID - 8; y += 6) {
      for (let x = 6; x < GRID - 6; x++) {
        if (rng() > 0.4) {
          grid[x][y] = true;
        }
      }
    }
  } else if (family === "random_obstacles") {
    const density = params.density ?? 0.2;
    const numObstacles = Math.floor(GRID * GRID * density);
    
    for (let i = 0; i < numObstacles; i++) {
      const x = 3 + Math.floor(rng() * (GRID - 6));
      const y = Math.floor(rng() * GRID);
      grid[x][y] = true;
    }
  }
  
  return grid;
}

function ri(rng: () => number, lo: number, hi: number): number {
  return Math.floor(rng() * (hi - lo + 1)) + lo;
}

export const GRID_SIZE = GRID;