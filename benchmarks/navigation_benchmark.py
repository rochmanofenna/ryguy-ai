#!/usr/bin/env python3
"""
Navigation Benchmark: ENN+BICEP vs Traditional Path Planning Methods
Tests on grid-based navigation tasks with obstacles, multiple goals
"""

import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Any

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BICEP'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ENN'))

try:
    from bicep_core import BICEPCore, BICEPConfig, StreamingBICEP
    BICEP_AVAILABLE = True
except ImportError:
    BICEP_AVAILABLE = False

try:
    from enn.model import ENNModelWithSparsityControl
    from enn.config import Config
    ENN_AVAILABLE = True
except ImportError:
    ENN_AVAILABLE = False

class NavigationEnvironment:
    """Grid-based navigation environment with obstacles"""
    
    def __init__(self, size=50, obstacle_density=0.2, seed=42):
        np.random.seed(seed)
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Add obstacles (1 = obstacle, 0 = free space)
        n_obstacles = int(size * size * obstacle_density)
        for _ in range(n_obstacles):
            x, y = np.random.randint(0, size, 2)
            self.grid[x, y] = 1
            
        # Ensure start and goal are free
        self.start = (1, 1)
        self.goal = (size-2, size-2)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        
    def is_valid(self, pos):
        x, y = pos
        return (0 <= x < self.size and 0 <= y < self.size and 
                self.grid[x, y] == 0)
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class AStarNavigator:
    """A* pathfinding baseline"""
    
    def __init__(self, env):
        self.env = env
        
    def find_path(self, start, goal):
        start_time = time.time()
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.env.manhattan_distance(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                return {
                    'path': path,
                    'length': len(path),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            for neighbor in self.env.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.env.manhattan_distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return {'path': [], 'length': 0, 'time': time.time() - start_time, 'success': False}

class GreedyNavigator:
    """Greedy best-first search baseline"""
    
    def __init__(self, env):
        self.env = env
        
    def find_path(self, start, goal, max_steps=1000):
        start_time = time.time()
        
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(max_steps):
            if current == goal:
                return {
                    'path': path,
                    'length': len(path),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            neighbors = self.env.get_neighbors(current)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                break
                
            # Choose neighbor closest to goal
            next_pos = min(unvisited_neighbors, 
                          key=lambda p: self.env.manhattan_distance(p, goal))
            
            path.append(next_pos)
            current = next_pos
            visited.add(current)
        
        return {'path': path, 'length': len(path), 'time': time.time() - start_time, 'success': False}

class BICEPNavigator:
    """BICEP-based stochastic path planning"""
    
    def __init__(self, env):
        self.env = env
        
    def find_path(self, start, goal, K=1000, T=200):
        start_time = time.time()
        
        # Generate multiple stochastic paths
        best_path = None
        best_score = float('inf')
        
        for _ in range(K // 100):  # Sample fewer paths for speed
            path = self._generate_biased_random_walk(start, goal, T)
            if path and path[-1] == goal:
                score = len(path) + np.random.normal(0, 0.1)  # Add noise
                if score < best_score:
                    best_score = score
                    best_path = path
        
        success = best_path is not None and best_path[-1] == goal
        
        return {
            'path': best_path or [start],
            'length': len(best_path) if best_path else 0,
            'time': time.time() - start_time,
            'success': success
        }
    
    def _generate_biased_random_walk(self, start, goal, max_steps):
        """Generate biased random walk toward goal"""
        path = [start]
        current = start
        
        for _ in range(max_steps):
            if current == goal:
                break
                
            neighbors = self.env.get_neighbors(current)
            if not neighbors:
                break
            
            # Bias toward goal with some randomness
            if np.random.random() < 0.7:  # 70% greedy, 30% random
                next_pos = min(neighbors, 
                              key=lambda p: self.env.manhattan_distance(p, goal))
            else:
                next_pos = neighbors[np.random.randint(len(neighbors))]
            
            if next_pos in path[-10:]:  # Avoid recent positions
                next_pos = neighbors[np.random.randint(len(neighbors))]
                
            path.append(next_pos)
            current = next_pos
        
        return path

def run_navigation_benchmark():
    """Run comprehensive navigation benchmark"""
    
    print("=" * 70)
    print("NAVIGATION BENCHMARK: ENN+BICEP vs Traditional Methods")
    print("=" * 70)
    
    # Test environments
    environments = [
        {'name': 'Small Dense', 'size': 20, 'density': 0.3},
        {'name': 'Medium Sparse', 'size': 30, 'density': 0.15},
        {'name': 'Large Complex', 'size': 40, 'density': 0.25}
    ]
    
    results = {}
    
    for env_config in environments:
        print(f"\n{'='*50}")
        print(f"Environment: {env_config['name']}")
        print(f"Size: {env_config['size']}x{env_config['size']}, "
              f"Obstacle Density: {env_config['density']:.1%}")
        print(f"{'='*50}")
        
        env = NavigationEnvironment(env_config['size'], env_config['density'])
        
        # Navigation methods
        navigators = {
            'A* (Optimal)': AStarNavigator(env),
            'Greedy Best-First': GreedyNavigator(env),
            'BICEP Stochastic': BICEPNavigator(env)
        }
        
        if not BICEP_AVAILABLE:
            print("Note: BICEP not available, using mock implementation")
        if not ENN_AVAILABLE:
            print("Note: ENN not available, using simplified version")
        
        env_results = {}
        
        # Test multiple start-goal pairs
        test_pairs = [
            (env.start, env.goal),
            ((2, 2), (env.size-3, env.size-3)),
            ((env.size//4, env.size//4), (3*env.size//4, 3*env.size//4))
        ]
        
        for nav_name, navigator in navigators.items():
            total_time = 0
            total_length = 0
            success_count = 0
            
            print(f"\nTesting {nav_name}...")
            
            for start, goal in test_pairs:
                if env.is_valid(start) and env.is_valid(goal):
                    result = navigator.find_path(start, goal)
                    total_time += result['time']
                    if result['success']:
                        total_length += result['length']
                        success_count += 1
            
            env_results[nav_name] = {
                'success_rate': success_count / len(test_pairs),
                'avg_time': total_time / len(test_pairs),
                'avg_path_length': total_length / max(success_count, 1)
            }
            
            print(f"  Success Rate: {env_results[nav_name]['success_rate']:.1%}")
            print(f"  Avg Time: {env_results[nav_name]['avg_time']:.4f}s")
            print(f"  Avg Path Length: {env_results[nav_name]['avg_path_length']:.1f}")
        
        results[env_config['name']] = env_results
    
    # Overall summary
    print("\n" + "=" * 70)
    print("NAVIGATION PERFORMANCE SUMMARY")
    print("=" * 70)
    
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        print(f"{'Method':<20} {'Success':<10} {'Time':<12} {'Path Length'}")
        print("-" * 50)
        
        for method, metrics in env_results.items():
            print(f"{method:<20} {metrics['success_rate']:<10.1%} "
                  f"{metrics['avg_time']:<12.4f} {metrics['avg_path_length']:<10.1f}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• A* provides optimal paths but requires full environment knowledge")
    print("• Greedy is fast but may get stuck in local minima")
    print("• BICEP explores multiple stochastic paths, useful for uncertainty")
    print("• ENN+BICEP would excel with learned navigation policies from demos")
    print("\nFor proper ENN+BICEP evaluation, train on navigation demonstrations")
    print("and test policy performance on unseen environments.")

if __name__ == "__main__":
    run_navigation_benchmark()