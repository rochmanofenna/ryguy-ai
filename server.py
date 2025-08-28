#!/usr/bin/env python3
"""
FastAPI server for BICEP/ENN integration with web demo
"""

import sys
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add BICEP and ENN to path
sys.path.append('./BICEP')
sys.path.append('./ENN')

from bicep_core import BICEPCore, StreamingBICEP, BICEPConfig
from enn import create_attention_enn
from enn.config import Config
from enn.bicep_layers import ENNBICEPHybrid, create_bicep_enhanced_model

app = FastAPI(title="BICEP/ENN Navigation Server")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
enn_model = None
bicep_core = None
hybrid_model = None

class DemoRequest(BaseModel):
    grid: List[List[bool]]  # [W][H] obstacle map
    start: List[List[int]]  # [[x,y], ...] start positions
    goal: List[List[int]]   # [[x,y], ...] goal positions
    K: int = 5000           # Number of paths
    T: int = 300            # Time horizon
    mu: float = 1.0         # Drift strength
    sigma: float = 0.8      # Noise scale
    seed: Optional[int] = None
    batch_size: int = 2000
    use_antithetic: bool = False
    correlation: float = 0.0

class ActRequest(BaseModel):
    grid: List[List[bool]]
    start: List[List[int]]
    goal: List[List[int]]
    state: Dict[str, float]  # {"x": float, "y": float}
    use_hybrid: bool = False

class DemoPath(BaseModel):
    path: List[List[float]]  # [[x,y], ...]
    cost: float
    risk: float
    clearance: float
    id: int

@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global enn_model, bicep_core, hybrid_model
    
    # Initialize ENN
    config = Config()
    config.num_neurons = 10
    config.num_states = 5
    enn_model = create_attention_enn(config, input_size=(4, 40, 40), num_actions=4)
    enn_model.eval()
    
    # Initialize BICEP
    bicep_config = BICEPConfig(
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        use_half_precision=False,  # Keep full precision for accuracy
        enable_memory_pooling=True
    )
    bicep_core = StreamingBICEP(bicep_config)
    
    # Initialize hybrid model
    hybrid_model = ENNBICEPHybrid(config, bicep_integration_mode='parallel')
    
    print(f"Models initialized - Device: {bicep_config.device}")

def grid_to_tensor(grid: List[List[bool]], start: List[List[int]], 
                   goal: List[List[int]], state: Dict[str, float]) -> torch.Tensor:
    """Convert grid state to 4-channel tensor for ENN input"""
    H = W = 40  # Fixed size for now
    C = 4  # channels: obstacles, start, goal, agent
    
    # Initialize tensor
    x = np.zeros((1, C, H, W), dtype=np.float32)
    
    # Fill obstacle channel
    grid_array = np.array(grid)
    if grid_array.shape[0] != W or grid_array.shape[1] != H:
        # Resize if needed
        from scipy.ndimage import zoom
        scale_w = W / grid_array.shape[0]
        scale_h = H / grid_array.shape[1]
        grid_array = zoom(grid_array.astype(float), (scale_w, scale_h), order=0) > 0.5
    
    x[0, 0] = grid_array.T  # Transpose for correct indexing
    
    # Fill start positions
    for sx, sy in start:
        if 0 <= sx < W and 0 <= sy < H:
            x[0, 1, sy, sx] = 1.0
    
    # Fill goal positions
    for gx, gy in goal:
        if 0 <= gx < W and 0 <= gy < H:
            x[0, 2, gy, gx] = 1.0
    
    # Fill agent position
    ax = int(state['x'])
    ay = int(state['y'])
    if 0 <= ax < W and 0 <= ay < H:
        x[0, 3, ay, ax] = 1.0
    
    return torch.from_numpy(x)

def calculate_path_risk(path: List[List[float]], grid: List[List[bool]]) -> Dict[str, float]:
    """Calculate risk metrics for a path"""
    total_risk = 0.0
    min_clearance = float('inf')
    
    for x, y in path:
        # Calculate minimum clearance to obstacles
        local_clearance = float('inf')
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                    
                gx = int(x + dx)
                gy = int(y + dy)
                
                if 0 <= gx < len(grid) and 0 <= gy < len(grid[0]) and grid[gx][gy]:
                    clearance = np.sqrt(dx*dx + dy*dy)
                    local_clearance = min(local_clearance, clearance)
        
        min_clearance = min(min_clearance, local_clearance)
        
        # Risk penalty: exponential in low clearance
        if local_clearance < 2.5:
            total_risk += np.exp(2.5 - local_clearance) * 0.1
    
    risk_score = total_risk + (2.0 if min_clearance < 1.5 else 0)
    
    return {
        'clearance_risk': total_risk,
        'min_clearance': min_clearance,
        'risk_score': risk_score
    }

@app.post("/bicep/demos", response_model=Dict[str, Any])
async def generate_demos(req: DemoRequest):
    """Generate demonstration paths using BICEP"""
    try:
        # Convert grid to numpy array
        grid_array = np.array(req.grid, dtype=bool)
        
        # Configure BICEP sampler
        bicep_core.configure(
            n_paths=req.K,
            n_steps=req.T,
            dt=1.0 / req.T,
            mu=req.mu,
            sigma=req.sigma,
            batch_size=req.batch_size,
            seed=req.seed
        )
        
        # Generate paths
        paths_raw, stats = bicep_core.stream_generate(
            grid=grid_array,
            start_positions=req.start,
            goal_positions=req.goal,
            use_antithetic=req.use_antithetic,
            correlation=req.correlation
        )
        
        # Process paths and compute metrics
        demo_paths = []
        for i, path in enumerate(paths_raw[:100]):  # Limit to top 100 paths
            path_list = path.tolist() if isinstance(path, np.ndarray) else path
            
            # Calculate cost (path length)
            cost = len(path_list) * (1.0 / req.T)
            
            # Calculate risk metrics
            risk_metrics = calculate_path_risk(path_list, req.grid)
            
            demo_paths.append({
                'path': path_list,
                'cost': cost,
                'risk': risk_metrics['risk_score'],
                'clearance': risk_metrics['min_clearance'],
                'id': i
            })
        
        # Sort by risk or cost
        demo_paths.sort(key=lambda x: x['risk'])
        
        return {
            'demos': demo_paths[:50],  # Return top 50
            'stats': {
                'total_generated': len(paths_raw),
                'success_rate': stats.get('success_rate', 0),
                'avg_cost': np.mean([p['cost'] for p in demo_paths[:50]]),
                'avg_risk': np.mean([p['risk'] for p in demo_paths[:50]])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BICEP generation failed: {str(e)}")

@app.post("/enn/act", response_model=Dict[str, Any])
async def enn_act(req: ActRequest):
    """Get action from ENN model"""
    try:
        # Convert state to tensor
        x = grid_to_tensor(req.grid, req.start, req.goal, req.state)
        
        # Get action from appropriate model
        with torch.no_grad():
            if req.use_hybrid and hybrid_model is not None:
                logits = hybrid_model(x).squeeze(0)
            else:
                logits = enn_model(x).squeeze(0)
            
            # Convert to numpy for response
            logits_np = logits.cpu().numpy()
            action = int(np.argmax(logits_np))
            
            # Action mapping: 0=right, 1=down, 2=left, 3=up
            action_probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()
        
        return {
            'action': action,
            'logits': logits_np.tolist(),
            'action_probs': action_probs,
            'action_name': ['right', 'down', 'left', 'up'][action]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ENN inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'models': {
            'enn': enn_model is not None,
            'bicep': bicep_core is not None,
            'hybrid': hybrid_model is not None
        },
        'device': str(next(enn_model.parameters()).device) if enn_model else 'none'
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)