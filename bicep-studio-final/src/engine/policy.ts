export type Head = { 
  W1: Float32Array; 
  b1: Float32Array; 
  W2: Float32Array; 
  b2: Float32Array 
};

export type Ensemble = { 
  heads: Head[]; 
  hidden: number; 
  inDim: number 
};

export async function loadPolicyFromFile(): Promise<Ensemble | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return resolve(null);
      
      try {
        const text = await file.text();
        const obj = JSON.parse(text);
        
        const heads = obj.heads.map((h: any) => ({
          W1: toF32(h.W1),
          b1: toF32(h.b1),
          W2: toF32(h.W2),
          b2: toF32(h.b2)
        }));
        
        const hidden = heads[0].b1.length;
        const inDim = heads[0].W1.length / hidden;
        
        resolve({ heads, hidden, inDim });
      } catch (error) {
        console.error('Failed to load policy:', error);
        resolve(null);
      }
    };
    
    input.click();
  });
}

function toF32(m: number[] | number[][]): Float32Array {
  const flat = Array.isArray(m[0]) ? (m as number[][]).flat() : m as number[];
  return Float32Array.from(flat);
}

export function featurize(x: number, y: number, grid: boolean[][], GRID = 40): Float32Array {
  const feat: number[] = [];
  
  // 5x5 local occupancy grid
  for (let xx = x - 2; xx <= x + 2; xx++) {
    for (let yy = y - 2; yy <= y + 2; yy++) {
      feat.push(xx >= 0 && xx < GRID && yy >= 0 && yy < GRID ? (grid[xx][yy] ? 1 : 0) : 1);
    }
  }
  
  // Normalized position
  feat.push(x / GRID, y / GRID);
  
  return Float32Array.from(feat);
}

export function act(ens: Ensemble, x: number, y: number, grid: boolean[][]): number {
  const feat = featurize(x, y, grid);
  const { heads, hidden, inDim } = ens;
  const acc = new Float32Array(4);
  
  for (const h of heads) {
    // Layer 1: input -> hidden (ReLU)
    const h1 = new Float32Array(hidden);
    for (let j = 0; j < hidden; j++) {
      let sum = 0;
      for (let i = 0; i < inDim; i++) {
        sum += feat[i] * h.W1[i * hidden + j];
      }
      h1[j] = Math.max(0, sum + h.b1[j]);
    }
    
    // Layer 2: hidden -> output (linear)
    const logits = new Float32Array(4);
    for (let k = 0; k < 4; k++) {
      let sum = 0;
      for (let j = 0; j < hidden; j++) {
        sum += h1[j] * h.W2[j * 4 + k];
      }
      logits[k] = sum + h.b2[k];
      acc[k] += logits[k];
    }
  }
  
  // Return argmax action
  let best = 0;
  for (let k = 1; k < 4; k++) {
    if (acc[k] > acc[best]) {
      best = k;
    }
  }
  
  return best; // 0:right, 1:down, 2:left, 3:up
}