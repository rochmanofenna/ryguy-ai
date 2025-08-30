#!/usr/bin/env python3
"""
Interactive ENN+BICEP Navigation Demos
Showcase unique advantages: uncertainty, adaptation, decision-making
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
from typing import List, Tuple, Dict, Any

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ENN'))

# Import our improved navigation system
from improved_enn_bicep_navigation import (
    EnhancedNavigationEnvironment, 
    ImprovedBICEPDemonstrationGenerator,
    ImprovedENNBICEPNavigator,
    EnhancedENN
)

class NavigationDemonstrator:
    """Interactive demos showing ENN+BICEP advantages"""
    
    def __init__(self):
        self.env = None
        self.enn_agent = None
        self.setup_agents()
        
    def setup_agents(self):
        """Setup environment and trained agent"""
        print("Setting up navigation demonstration environment...")
        
        # Create environment
        self.env = EnhancedNavigationEnvironment(size=25, obstacle_density=0.2, seed=123)
        
        # Quick training for demo purposes
        demo_generator = ImprovedBICEPDemonstrationGenerator(self.env)
        features, actions = demo_generator.generate_comprehensive_dataset(num_demonstrations=300)
        
        # Train ENN agent
        self.enn_agent = ImprovedENNBICEPNavigator(self.env)
        self.enn_agent.train((features, actions), epochs=50, batch_size=64)
        
        print("Demo setup complete!")
    
    def demo_1_uncertainty_quantification(self):
        """Demo 1: Show how ENN+BICEP quantifies decision uncertainty"""
        print("\n" + "="*70)
        print("DEMO 1: UNCERTAINTY QUANTIFICATION")
        print("="*70)
        print("ENN+BICEP provides confidence scores for navigation decisions")
        print("High confidence = clear path, Low confidence = complex decisions")
        
        # Test scenarios with different complexity levels
        scenarios = [
            {"name": "Clear Path", "start": (2, 2), "goal": (22, 22), "expected": "High Confidence"},
            {"name": "Narrow Corridor", "start": (2, 12), "goal": (22, 12), "expected": "Medium Confidence"},  
            {"name": "Complex Maze", "start": (2, 2), "goal": (12, 22), "expected": "Variable Confidence"}
        ]
        
        for scenario in scenarios:
            start, goal = scenario["start"], scenario["goal"]
            if self.env.is_valid(start) and self.env.is_valid(goal):
                result = self.enn_agent.find_path(start, goal)
                
                confidence_level = "High" if result.get('confidence', 0) > 0.8 else \
                                 "Medium" if result.get('confidence', 0) > 0.6 else "Low"
                
                print(f"\n{scenario['name']:15} | Success: {result['success']} | "
                      f"Confidence: {result.get('confidence', 0):.2f} ({confidence_level})")
                print(f"Expected: {scenario['expected']}")
        
        print("\n💡 Key Insight: Traditional A* gives no confidence measure!")
        print("   ENN+BICEP tells you HOW confident it is in each decision.")
    
    def demo_2_dynamic_adaptation(self):
        """Demo 2: Adaptation to dynamic environments"""
        print("\n" + "="*70)
        print("DEMO 2: DYNAMIC ENVIRONMENT ADAPTATION")
        print("="*70)
        print("ENN+BICEP adapts when environment changes during navigation")
        
        # Create base path
        start, goal = (2, 2), (22, 22)
        
        print(f"\nOriginal environment navigation: {start} -> {goal}")
        original_result = self.enn_agent.find_path(start, goal)
        print(f"Original path length: {original_result['length']} steps")
        
        # Simulate dynamic obstacle appearance
        print("\n🚧 SIMULATION: New obstacles appear during navigation!")
        
        # Add obstacles in middle of environment
        original_grid = self.env.grid.copy()
        for x in range(10, 15):
            for y in range(10, 15):
                if self.env.is_valid((x, y)):
                    self.env.grid[x, y] = 1
        
        adapted_result = self.enn_agent.find_path(start, goal)
        print(f"Adapted path length: {adapted_result['length']} steps")
        print(f"Adaptation success: {adapted_result['success']}")
        
        # Restore environment
        self.env.grid = original_grid
        
        print(f"\n💡 Key Insight: A* would need complete re-planning!")
        print("   ENN+BICEP adapts incrementally using learned spatial reasoning.")
    
    def demo_3_decision_visualization(self):
        """Demo 3: Visualize decision-making process"""
        print("\n" + "="*70)
        print("DEMO 3: DECISION-MAKING VISUALIZATION")
        print("="*70)
        print("Comparing navigation strategies across different methods")
        
        # Import baseline methods
        from navigation_benchmark import AStarNavigator, BICEPNavigator
        
        start, goal = (2, 2), (20, 20)
        agents = {
            "A* (Optimal)": AStarNavigator(self.env),
            "BICEP (Stochastic)": BICEPNavigator(self.env), 
            "ENN+BICEP (Learned)": self.enn_agent
        }
        
        print(f"\nNavigation comparison: {start} -> {goal}")
        print("-" * 50)
        
        results = {}
        for name, agent in agents.items():
            result = agent.find_path(start, goal)
            results[name] = result
            
            path_type = "Optimal" if "A*" in name else \
                       "Exploratory" if "BICEP" in name and "ENN" not in name else \
                       "Learned"
            
            confidence_str = f", Confidence: {result.get('confidence', 0):.2f}" \
                           if 'confidence' in result else ""
            
            print(f"{name:20} | Length: {result['length']:3d} | Type: {path_type:11} | "
                  f"Success: {result['success']}{confidence_str}")
        
        print(f"\n💡 Key Insights:")
        print("   • A* finds shortest path but needs complete map knowledge")
        print("   • BICEP explores multiple paths but can be inefficient")  
        print("   • ENN+BICEP balances efficiency with learned spatial understanding")
    
    def demo_4_feature_importance(self):
        """Demo 4: Show what features ENN+BICEP focuses on"""
        print("\n" + "="*70)
        print("DEMO 4: SPATIAL FEATURE ANALYSIS")
        print("="*70)
        print("Understanding what ENN+BICEP 'sees' in the environment")
        
        start, goal = (2, 2), (20, 20)
        
        # Extract features at different positions along a path
        test_positions = [(2, 2), (5, 5), (10, 10), (15, 15), (20, 20)]
        
        print(f"\nFeature analysis from {start} to {goal}:")
        print("-" * 60)
        
        for i, pos in enumerate(test_positions):
            if self.env.is_valid(pos):
                features = self.env.extract_enhanced_features(pos, goal, [start, pos])
                
                # Key feature indices
                goal_distance = features[7]  # Euclidean distance to goal
                obstacle_density = features[21]  # Local obstacle density
                path_clarity = features[17]  # Direct path clearance (right direction)
                progress = features[22]  # Progress toward goal
                
                print(f"Position {str(pos):8} | Distance: {goal_distance:.2f} | "
                      f"Obstacles: {obstacle_density:.2f} | "
                      f"Path Clear: {path_clarity:.2f} | Progress: {progress:.2f}")
        
        print(f"\n💡 Key Insight: ENN+BICEP processes 27 spatial features simultaneously!")
        print("   Traditional methods use simple distance heuristics.")
    
    def demo_5_ensemble_uncertainty(self):
        """Demo 5: Show ensemble decision-making"""
        print("\n" + "="*70)
        print("DEMO 5: ENSEMBLE DECISION MAKING")
        print("="*70)
        print("ENN+BICEP uses 5 neural network 'experts' for each decision")
        
        # Test at a complex decision point
        complex_position = (12, 12)  # Middle of environment
        goal = (20, 20)
        
        if self.env.is_valid(complex_position):
            # Get model predictions
            features = self.env.extract_enhanced_features(complex_position, goal)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                main_output, ensemble_outputs = self.enn_agent.model(features_tensor)
                
                print(f"\nEnsemble analysis at position {complex_position}:")
                print("Action probabilities from 5 expert networks:")
                print("-" * 50)
                
                actions = ["Right", "Down", "Left", "Up"]
                
                for i, expert_output in enumerate(ensemble_outputs):
                    probs = torch.softmax(expert_output, dim=1)[0]
                    best_action = actions[torch.argmax(probs)]
                    confidence = torch.max(probs).item()
                    
                    print(f"Expert {i+1}: {best_action:5} (confidence: {confidence:.2f})")
                
                # Final ensemble decision
                final_probs = torch.softmax(main_output, dim=1)[0]
                final_action = actions[torch.argmax(final_probs)]
                final_confidence = torch.max(final_probs).item()
                
                print("-" * 50)
                print(f"Final Decision: {final_action} (confidence: {final_confidence:.2f})")
        
        print(f"\n💡 Key Insight: Multiple experts provide robust decision-making!")
        print("   Single networks can be overconfident or biased.")
    
    def create_visual_demo(self, save_path="navigation_demo.png"):
        """Create visual comparison of different navigation approaches"""
        print(f"\n" + "="*70)
        print("CREATING VISUAL DEMO")
        print("="*70)
        
        from navigation_benchmark import AStarNavigator, BICEPNavigator
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Navigation Methods Comparison', fontsize=16, fontweight='bold')
        
        start, goal = (2, 2), (22, 22)
        
        # Method configurations
        methods = [
            ("A* (Optimal)", AStarNavigator(self.env), axes[0, 0]),
            ("BICEP (Stochastic)", BICEPNavigator(self.env), axes[0, 1]), 
            ("ENN+BICEP (Learned)", self.enn_agent, axes[1, 0])
        ]
        
        # Environment visualization on last subplot
        self.visualize_environment(axes[1, 1])
        
        for method_name, agent, ax in methods:
            result = agent.find_path(start, goal)
            self.visualize_path(ax, result['path'], start, goal, method_name, result)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visual demo saved to: {save_path}")
        
    def visualize_environment(self, ax):
        """Visualize the navigation environment"""
        ax.imshow(self.env.grid, cmap='Greys', origin='lower')
        ax.set_title('Environment Layout')
        ax.set_xlabel('X Position')  
        ax.set_ylabel('Y Position')
        
        # Mark start and goal positions
        for start in self.env.start_positions[:3]:
            ax.plot(start[1], start[0], 'go', markersize=8, label='Start' if start == self.env.start_positions[0] else "")
        for goal in self.env.goal_positions[:3]:
            ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal' if goal == self.env.goal_positions[0] else "")
        
        ax.legend()
    
    def visualize_path(self, ax, path, start, goal, title, result):
        """Visualize a navigation path"""
        ax.imshow(self.env.grid, cmap='Greys', origin='lower', alpha=0.7)
        
        if path and len(path) > 1:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.8)
            ax.plot(path_x, path_y, 'bo', markersize=3, alpha=0.6)
        
        # Mark start and goal
        ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
        ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
        
        # Add result info
        success_emoji = "✓" if result['success'] else "✗"
        confidence_str = f", Conf: {result.get('confidence', 0):.2f}" if 'confidence' in result else ""
        info = f"{success_emoji} {len(path)} steps{confidence_str}"
        
        ax.set_title(f'{title}\n{info}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    def run_all_demos(self):
        """Run all demonstration scenarios"""
        print("🚀 ENN+BICEP NAVIGATION DEMONSTRATIONS")
        print("=" * 70)
        print("Showcasing unique advantages of learned navigation")
        
        self.demo_1_uncertainty_quantification()
        self.demo_2_dynamic_adaptation()
        self.demo_3_decision_visualization()
        self.demo_4_feature_importance()
        self.demo_5_ensemble_uncertainty()
        
        print(f"\n" + "="*70)
        print("SUMMARY: ENN+BICEP ADVANTAGES")
        print("="*70)
        print("✅ Uncertainty Quantification - Knows when decisions are risky")
        print("✅ Dynamic Adaptation - Adjusts to environment changes")
        print("✅ Learned Spatial Reasoning - 27-dimensional feature processing") 
        print("✅ Ensemble Robustness - Multiple experts vote on decisions")
        print("✅ Temporal Context - Uses path history for better decisions")
        print("✅ Confidence Estimation - Provides reliability scores")
        
        print(f"\n💡 WHEN TO USE ENN+BICEP:")
        print("• Dynamic environments (obstacles change)")
        print("• Uncertain conditions (partial observability)")
        print("• Risk assessment needed (confidence matters)")
        print("• Learning from demonstrations available")
        print("• Complex spatial reasoning required")
        
        print(f"\n💡 WHEN TO USE A*:")
        print("• Static environments (obstacles fixed)")
        print("• Complete map knowledge available")
        print("• Optimal path length critical")
        print("• Computational resources limited")
        
        # Create visual demo
        self.create_visual_demo()
        
        print(f"\n🎯 Ready for interactive web demo at http://localhost:5173")

def main():
    """Run the demonstration suite"""
    demonstrator = NavigationDemonstrator()
    demonstrator.run_all_demos()

if __name__ == "__main__":
    main()