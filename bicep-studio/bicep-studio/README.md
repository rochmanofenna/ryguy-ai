# BICEP Studio

A modern web application for demonstrating BICEP (Brownian-Inspired CUDA Engine for Paths) stochastic path sampling and ENN (Entangled Neural Networks) policy evaluation.

## Features

- **Real-time BICEP simulation** with WebWorker-based path sampling
- **Multiple map families**: Corridors, Rooms & Doors, Mazes, Random Obstacles  
- **ENN policy evaluation** with ensemble inference
- **Deterministic replays** via seeded RNG
- **Export capabilities** (PNG images, shareable URLs)
- **Modern UI** built with React, TypeScript, and Tailwind CSS

## Quick Start

```bash
npm install
npm run dev
```

Then open http://localhost:5173

## Usage

1. **Generate Map**: Choose a map family and click "New Map" or "Apply Changes"
2. **Run BICEP**: Configure sampler parameters (K, T, μ, σ) and click "Run BICEP" 
3. **Load Policy**: Click "Load Policy" to import a trained ENN ensemble JSON file
4. **Evaluate**: Test the loaded policy on the current map with "Evaluate"
5. **Share**: Copy deterministic replay links or export visualization PNGs

## File Structure

```
src/
├── app/BICEPStudio.tsx     # Main application component
├── engine/
│   ├── types.ts            # Shared type definitions  
│   ├── mapFactory.ts       # Map generation algorithms
│   ├── bicep.worker.ts     # WebWorker BICEP sampler
│   └── policy.ts           # ENN policy loading/inference
├── lib/
│   ├── rng.ts             # Deterministic RNG utilities
│   └── seedUrl.ts         # URL state serialization
└── ui/                    # Reusable UI components
```

## Test Policy

Generate a test policy file:

```bash
python3 generate_test_policy.py
```

This creates `test_policy.json` with a 5-head ensemble (27→128→4 architecture) that you can load in the application.

## Technology Stack

- **Frontend**: React 18, TypeScript, Vite
- **Styling**: Tailwind CSS with custom variables  
- **Icons**: Lucide React
- **WebWorkers**: ES modules for background simulation
- **Canvas**: Multi-layer rendering for visualization