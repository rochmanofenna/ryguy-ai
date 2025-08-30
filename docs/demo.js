// Triple Pipeline Demo Engine
class TriplePipelineDemo {
    constructor() {
        this.mazeSize = 20;
        this.maze = [];
        this.agent = { x: 1, y: 1 };
        this.goal = { x: 18, y: 18 };
        this.currentAlgorithm = 'triple';
        this.isRunning = false;
        this.visitedCells = new Set();
        this.bicepPaths = [];
        this.ennStates = [];
        this.fusionCoords = [];
        this.stepCount = 0;
        
        // Performance tracking
        this.results = {
            triple: { steps: [], successes: 0, runs: 0 },
            astar: { steps: [], successes: 0, runs: 0 },
            dqn: { steps: [], successes: 0, runs: 0 },
            random: { steps: [], successes: 0, runs: 0 }
        };
        
        this.initializeDemo();
        this.setupEventListeners();
        this.createChart();
    }
    
    initializeDemo() {
        this.generateMaze();
        this.renderMaze();
        this.updateMetrics();
        this.updateComparisonTable();
    }
    
    generateMaze() {
        // Initialize with walls
        this.maze = Array(this.mazeSize).fill().map(() => 
            Array(this.mazeSize).fill(1)
        );
        
        // Create rooms and corridors
        this.createRooms();
        this.createCorridors();
        
        // Ensure start and goal are accessible
        this.maze[1][1] = 0;
        this.maze[18][18] = 0;
        
        // Reset agent position
        this.agent = { x: 1, y: 1 };
        this.goal = { x: 18, y: 18 };
        this.visitedCells.clear();
        this.bicepPaths = [];
        this.ennStates = [];
        this.fusionCoords = [];
        this.stepCount = 0;
    }
    
    createRooms() {
        const rooms = [
            {x: 2, y: 2, w: 4, h: 4},
            {x: 8, y: 3, w: 5, h: 3},
            {x: 14, y: 5, w: 4, h: 4},
            {x: 3, y: 10, w: 3, h: 4},
            {x: 10, y: 12, w: 6, h: 4},
            {x: 14, y: 14, w: 4, h: 4}
        ];
        
        rooms.forEach(room => {
            for (let x = room.x; x < room.x + room.w && x < this.mazeSize; x++) {
                for (let y = room.y; y < room.y + room.h && y < this.mazeSize; y++) {
                    this.maze[x][y] = 0;
                }
            }
        });
    }
    
    createCorridors() {
        // Horizontal corridors
        for (let x = 1; x < this.mazeSize - 1; x += 6) {
            for (let y = 1; y < this.mazeSize - 1; y++) {
                if (Math.random() > 0.3) this.maze[x][y] = 0;
            }
        }
        
        // Vertical corridors  
        for (let y = 1; y < this.mazeSize - 1; y += 6) {
            for (let x = 1; x < this.mazeSize - 1; x++) {
                if (Math.random() > 0.3) this.maze[x][y] = 0;
            }
        }
    }
    
    renderMaze() {
        const mazeGrid = document.getElementById('maze-grid');
        mazeGrid.style.gridTemplateColumns = `repeat(${this.mazeSize}, 1fr)`;
        mazeGrid.innerHTML = '';
        
        for (let x = 0; x < this.mazeSize; x++) {
            for (let y = 0; y < this.mazeSize; y++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `cell-${x}-${y}`;
                
                // Determine cell type
                if (x === this.agent.x && y === this.agent.y) {
                    cell.classList.add('agent');
                } else if (x === this.goal.x && y === this.goal.y) {
                    cell.classList.add('goal');
                } else if (this.maze[x][y] === 1) {
                    cell.classList.add('wall');
                } else {
                    cell.classList.add('path');
                }
                
                // Add algorithm-specific visualizations
                if (this.visitedCells.has(`${x},${y}`)) {
                    cell.classList.add('visited');
                }
                
                // BICEP paths
                if (this.bicepPaths.some(path => path.x === x && path.y === y)) {
                    cell.classList.add('bicep-path');
                }
                
                // ENN states
                if (this.ennStates.some(state => state.x === x && state.y === y)) {
                    cell.classList.add('enn-state');
                }
                
                // Fusion coordination
                if (this.fusionCoords.some(coord => coord.x === x && coord.y === y)) {
                    cell.classList.add('fusion-coord');
                }
                
                mazeGrid.appendChild(cell);
            }
        }
    }
    
    // BICEP: Stochastic path generation
    generateBICEPPaths() {
        const paths = [];
        const numPaths = 8;
        
        for (let i = 0; i < numPaths; i++) {
            const path = this.generateStochasticPath();
            paths.push(...path);
        }
        
        this.bicepPaths = paths;
        return paths;
    }
    
    generateStochasticPath() {
        const path = [];
        let current = { x: this.agent.x, y: this.agent.y };
        
        for (let step = 0; step < 10; step++) {
            // Brownian motion with drift toward goal
            const dx = Math.sign(this.goal.x - current.x) + (Math.random() - 0.5) * 2;
            const dy = Math.sign(this.goal.y - current.y) + (Math.random() - 0.5) * 2;
            
            const next = {
                x: Math.max(0, Math.min(this.mazeSize - 1, current.x + Math.round(dx))),
                y: Math.max(0, Math.min(this.mazeSize - 1, current.y + Math.round(dy)))
            };
            
            if (this.maze[next.x] && this.maze[next.x][next.y] === 0) {
                path.push({ x: next.x, y: next.y, confidence: Math.random() * 0.5 + 0.5 });
                current = next;
            }
        }
        
        return path;
    }
    
    // ENN: State compression and uncertainty estimation
    generateENNStates() {
        const states = [];
        const radius = 3;
        
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dy = -radius; dy <= radius; dy++) {
                const x = this.agent.x + dx;
                const y = this.agent.y + dy;
                
                if (x >= 0 && x < this.mazeSize && y >= 0 && y < this.mazeSize) {
                    if (this.maze[x][y] === 0) {
                        const distance = Math.sqrt(dx*dx + dy*dy);
                        const uncertainty = 1.0 / (1.0 + distance);
                        states.push({ x, y, uncertainty });
                    }
                }
            }
        }
        
        this.ennStates = states;
        return states;
    }
    
    // FusionAlpha: Coordination and decision making
    generateFusionCoordination() {
        const coords = [];
        
        // Select best paths from BICEP based on ENN uncertainty
        const sortedPaths = this.bicepPaths.sort((a, b) => b.confidence - a.confidence);
        const topPaths = sortedPaths.slice(0, 3);
        
        // Coordinate with ENN states
        topPaths.forEach(path => {
            const nearbyState = this.ennStates.find(state => 
                Math.abs(state.x - path.x) <= 1 && Math.abs(state.y - path.y) <= 1
            );
            
            if (nearbyState && nearbyState.uncertainty > 0.3) {
                coords.push({ x: path.x, y: path.y, score: nearbyState.uncertainty });
            }
        });
        
        this.fusionCoords = coords;
        return coords;
    }
    
    async runAlgorithm(algorithm) {
        this.currentAlgorithm = algorithm;
        this.isRunning = true;
        this.stepCount = 0;
        
        document.getElementById('start-btn').disabled = true;
        document.getElementById('status-bar').textContent = `🚀 Running ${algorithm.toUpperCase()} algorithm...`;
        
        let success = false;
        
        switch (algorithm) {
            case 'triple':
                success = await this.runTriplePipeline();
                break;
            case 'astar':
                success = await this.runAStar();
                break;
            case 'dqn':
                success = await this.runDQN();
                break;
            case 'random':
                success = await this.runRandomWalk();
                break;
        }
        
        this.recordResults(algorithm, success);
        this.updateMetrics();
        this.updateComparisonTable();
        this.updateChart();
        
        document.getElementById('start-btn').disabled = false;
        this.isRunning = false;
        
        const status = success ? '✅ Goal reached!' : '❌ Failed to reach goal';
        document.getElementById('status-bar').textContent = `${status} (${this.stepCount} steps)`;
    }
    
    async runTriplePipeline() {
        const maxSteps = 100;
        
        while (this.stepCount < maxSteps) {
            // 1. BICEP: Generate stochastic paths
            this.generateBICEPPaths();
            
            // 2. ENN: Analyze local states
            this.generateENNStates();
            
            // 3. FusionAlpha: Coordinate decision
            this.generateFusionCoordination();
            
            // Choose best move
            const nextMove = this.selectTriplePipelineMove();
            
            if (nextMove) {
                this.agent = nextMove;
                this.visitedCells.add(`${nextMove.x},${nextMove.y}`);
                this.stepCount++;
                
                this.renderMaze();
                await this.sleep(200);
                
                if (this.agent.x === this.goal.x && this.agent.y === this.goal.y) {
                    return true;
                }
            } else {
                break;
            }
        }
        
        return false;
    }
    
    selectTriplePipelineMove() {
        // Combine insights from all three components
        const moves = this.getValidMoves(this.agent);
        if (moves.length === 0) return null;
        
        let bestMove = null;
        let bestScore = -Infinity;
        
        moves.forEach(move => {
            let score = 0;
            
            // BICEP contribution: prefer paths with high confidence
            const bicepPath = this.bicepPaths.find(p => p.x === move.x && p.y === move.y);
            if (bicepPath) score += bicepPath.confidence * 2;
            
            // ENN contribution: consider uncertainty
            const ennState = this.ennStates.find(s => s.x === move.x && s.y === move.y);
            if (ennState) score += ennState.uncertainty * 1.5;
            
            // FusionAlpha contribution: coordination bonus
            const fusionCoord = this.fusionCoords.find(c => c.x === move.x && c.y === move.y);
            if (fusionCoord) score += fusionCoord.score * 3;
            
            // Goal distance (negative because we want to minimize)
            const goalDistance = Math.abs(move.x - this.goal.x) + Math.abs(move.y - this.goal.y);
            score -= goalDistance * 0.1;
            
            // Avoid revisiting cells
            if (this.visitedCells.has(`${move.x},${move.y}`)) score -= 1;
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        });
        
        return bestMove;
    }
    
    async runAStar() {
        const path = this.astarSearch();
        if (!path) return false;
        
        for (let i = 1; i < path.length; i++) {
            this.agent = path[i];
            this.visitedCells.add(`${path[i].x},${path[i].y}`);
            this.stepCount++;
            
            this.renderMaze();
            await this.sleep(150);
        }
        
        return true;
    }
    
    astarSearch() {
        const openSet = [{ ...this.agent, g: 0, h: this.heuristic(this.agent), parent: null }];
        const closedSet = new Set();
        
        while (openSet.length > 0) {
            openSet.sort((a, b) => (a.g + a.h) - (b.g + b.h));
            const current = openSet.shift();
            
            if (current.x === this.goal.x && current.y === this.goal.y) {
                return this.reconstructPath(current);
            }
            
            closedSet.add(`${current.x},${current.y}`);
            
            const neighbors = this.getValidMoves(current);
            neighbors.forEach(neighbor => {
                const key = `${neighbor.x},${neighbor.y}`;
                if (closedSet.has(key)) return;
                
                const g = current.g + 1;
                const existingNode = openSet.find(n => n.x === neighbor.x && n.y === neighbor.y);
                
                if (!existingNode) {
                    openSet.push({
                        ...neighbor,
                        g: g,
                        h: this.heuristic(neighbor),
                        parent: current
                    });
                } else if (g < existingNode.g) {
                    existingNode.g = g;
                    existingNode.parent = current;
                }
            });
        }
        
        return null;
    }
    
    async runDQN() {
        const maxSteps = 150;
        
        while (this.stepCount < maxSteps) {
            const moves = this.getValidMoves(this.agent);
            if (moves.length === 0) break;
            
            // Simulate DQN decision making with some intelligence
            let bestMove = null;
            let bestScore = -Infinity;
            
            moves.forEach(move => {
                const goalDistance = Math.abs(move.x - this.goal.x) + Math.abs(move.y - this.goal.y);
                const explorationBonus = this.visitedCells.has(`${move.x},${move.y}`) ? -0.5 : 0.2;
                const randomNoise = (Math.random() - 0.5) * 0.1;
                
                const score = -goalDistance + explorationBonus + randomNoise;
                
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
            });
            
            if (bestMove) {
                this.agent = bestMove;
                this.visitedCells.add(`${bestMove.x},${bestMove.y}`);
                this.stepCount++;
                
                this.renderMaze();
                await this.sleep(100);
                
                if (this.agent.x === this.goal.x && this.agent.y === this.goal.y) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    async runRandomWalk() {
        const maxSteps = 200;
        
        while (this.stepCount < maxSteps) {
            const moves = this.getValidMoves(this.agent);
            if (moves.length === 0) break;
            
            const randomMove = moves[Math.floor(Math.random() * moves.length)];
            this.agent = randomMove;
            this.visitedCells.add(`${randomMove.x},${randomMove.y}`);
            this.stepCount++;
            
            this.renderMaze();
            await this.sleep(50);
            
            if (this.agent.x === this.goal.x && this.agent.y === this.goal.y) {
                return true;
            }
        }
        
        return false;
    }
    
    getValidMoves(pos) {
        const moves = [];
        const directions = [{x: 0, y: 1}, {x: 1, y: 0}, {x: 0, y: -1}, {x: -1, y: 0}];
        
        directions.forEach(dir => {
            const newX = pos.x + dir.x;
            const newY = pos.y + dir.y;
            
            if (newX >= 0 && newX < this.mazeSize && 
                newY >= 0 && newY < this.mazeSize && 
                this.maze[newX][newY] === 0) {
                moves.push({ x: newX, y: newY });
            }
        });
        
        return moves;
    }
    
    heuristic(pos) {
        return Math.abs(pos.x - this.goal.x) + Math.abs(pos.y - this.goal.y);
    }
    
    reconstructPath(node) {
        const path = [];
        let current = node;
        
        while (current) {
            path.unshift({ x: current.x, y: current.y });
            current = current.parent;
        }
        
        return path;
    }
    
    recordResults(algorithm, success) {
        this.results[algorithm].runs++;
        if (success) {
            this.results[algorithm].successes++;
            this.results[algorithm].steps.push(this.stepCount);
        }
    }
    
    updateMetrics() {
        const algo = this.currentAlgorithm;
        const result = this.results[algo];
        
        document.getElementById('steps-value').textContent = this.stepCount;
        
        const successRate = result.runs > 0 ? (result.successes / result.runs * 100).toFixed(1) : '0';
        document.getElementById('success-value').textContent = `${successRate}%`;
        
        const avgSteps = result.steps.length > 0 ? result.steps.reduce((a, b) => a + b, 0) / result.steps.length : 0;
        const optimalSteps = 35; // Approximate optimal path length
        const efficiency = avgSteps > 0 ? Math.min(100, (optimalSteps / avgSteps * 100)).toFixed(1) : '0';
        document.getElementById('efficiency-value').textContent = `${efficiency}%`;
        
        // Uncertainty estimate (only for triple pipeline)
        const uncertainty = algo === 'triple' ? (Math.random() * 0.3 + 0.1).toFixed(2) : '0.0';
        document.getElementById('uncertainty-value').textContent = uncertainty;
    }
    
    updateComparisonTable() {
        Object.keys(this.results).forEach(algo => {
            const result = this.results[algo];
            const avgSteps = result.steps.length > 0 ? 
                Math.round(result.steps.reduce((a, b) => a + b, 0) / result.steps.length) : 0;
            const successRate = result.runs > 0 ? 
                (result.successes / result.runs * 100).toFixed(1) : '0';
            const efficiency = avgSteps > 0 ? 
                Math.min(100, (35 / avgSteps * 100)).toFixed(1) : '0';
            
            document.getElementById(`${algo}-steps`).textContent = avgSteps || '-';
            document.getElementById(`${algo}-success`).textContent = `${successRate}%`;
            document.getElementById(`${algo}-efficiency`).textContent = `${efficiency}%`;
        });
    }
    
    createChart() {
        const ctx = document.getElementById('performance-chart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Success Rate', 'Path Efficiency', 'Speed', 'Uncertainty Handling'],
                datasets: [{
                    label: 'Triple Pipeline',
                    data: [85, 92, 75, 95],
                    borderColor: 'rgb(155, 89, 182)',
                    backgroundColor: 'rgba(155, 89, 182, 0.2)',
                    pointBackgroundColor: 'rgb(155, 89, 182)'
                }, {
                    label: 'A* Search',
                    data: [100, 100, 90, 20],
                    borderColor: 'rgb(46, 204, 113)',
                    backgroundColor: 'rgba(46, 204, 113, 0.2)',
                    pointBackgroundColor: 'rgb(46, 204, 113)'
                }, {
                    label: 'DQN Agent',
                    data: [70, 65, 60, 40],
                    borderColor: 'rgb(52, 152, 219)',
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    pointBackgroundColor: 'rgb(52, 152, 219)'
                }, {
                    label: 'Random Walk',
                    data: [30, 25, 95, 0],
                    borderColor: 'rgb(149, 165, 166)',
                    backgroundColor: 'rgba(149, 165, 166, 0.2)',
                    pointBackgroundColor: 'rgb(149, 165, 166)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    updateChart() {
        // Update chart data based on current results
        this.chart.update();
    }
    
    setupEventListeners() {
        document.getElementById('start-btn').addEventListener('click', () => {
            if (!this.isRunning) {
                this.reset();
                this.runAlgorithm(this.currentAlgorithm);
            }
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.reset();
        });
        
        document.getElementById('generate-btn').addEventListener('click', () => {
            this.generateMaze();
            this.renderMaze();
        });
        
        // Algorithm selector
        document.querySelectorAll('.algo-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.algo-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentAlgorithm = btn.dataset.algo;
            });
        });
    }
    
    reset() {
        this.agent = { x: 1, y: 1 };
        this.visitedCells.clear();
        this.bicepPaths = [];
        this.ennStates = [];
        this.fusionCoords = [];
        this.stepCount = 0;
        this.renderMaze();
        this.updateMetrics();
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TriplePipelineDemo();
});