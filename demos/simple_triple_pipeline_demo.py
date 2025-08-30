#!/usr/bin/env python3
"""
Simplified Triple Pipeline Demo
Shows how BICEP + ENN + FusionAlpha work together
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches

def demonstrate_triple_pipeline():
    """Visual demonstration of the triple pipeline"""
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # BICEP: Stochastic Path Generation
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("BICEP: Stochastic Path Exploration", fontsize=14, fontweight='bold')
    
    # Generate multiple Brownian paths
    n_paths = 20
    steps = 50
    paths = []
    
    for i in range(n_paths):
        path = np.zeros((steps, 2))
        path[0] = [10, 10]  # Start position
        
        for t in range(1, steps):
            # Brownian motion with drift toward goal
            drift = np.array([20, 20]) - path[t-1]
            drift = drift / np.linalg.norm(drift) * 0.3
            noise = np.random.normal(0, 0.5, 2)
            path[t] = path[t-1] + drift + noise
            
        paths.append(path)
        alpha = 0.3 if i < n_paths - 1 else 1.0
        color = 'lightblue' if i < n_paths - 1 else 'darkblue'
        ax1.plot(path[:, 0], path[:, 1], alpha=alpha, color=color, linewidth=1)
    
    # Add start and goal
    ax1.scatter(10, 10, s=200, c='green', marker='s', label='Start', zorder=5)
    ax1.scatter(20, 20, s=200, c='red', marker='*', label='Goal', zorder=5)
    
    # Add obstacles
    obstacles = [(15, 12), (13, 17), (18, 15)]
    for ox, oy in obstacles:
        circle = Circle((ox, oy), 1, color='gray', alpha=0.7)
        ax1.add_patch(circle)
    
    ax1.set_xlim(5, 25)
    ax1.set_ylim(5, 25)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, "Generates multiple possible paths through uncertainty", 
             transform=ax1.transAxes, va='top', fontsize=10)
    
    # ENN: State Compression
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("ENN: Multi-State Compression", fontsize=14, fontweight='bold')
    
    # Visualize entangled states
    n_states = 5
    state_positions = np.random.rand(n_states, 2) * 0.8 + 0.1
    
    # Draw states
    for i, (x, y) in enumerate(state_positions):
        circle = Circle((x, y), 0.08, color=f'C{i}', alpha=0.6)
        ax2.add_patch(circle)
        ax2.text(x, y, f'S{i}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw entanglement connections
    for i in range(n_states):
        for j in range(i+1, n_states):
            if np.random.rand() > 0.5:
                ax2.plot([state_positions[i, 0], state_positions[j, 0]], 
                        [state_positions[i, 1], state_positions[j, 1]], 
                        'k-', alpha=0.2, linewidth=1)
    
    # Show collapse
    ax2.arrow(0.5, -0.05, 0, -0.1, head_width=0.05, head_length=0.03, 
              fc='black', ec='black', transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.text(0.5, 0.05, "Entangled States", ha='center', transform=ax2.transAxes)
    
    # ENN Output: Compressed representation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Collapsed Latent State", fontsize=14, fontweight='bold')
    
    # Show compressed state
    latent = np.random.randn(8, 8)
    im = ax3.imshow(latent, cmap='coolwarm', aspect='auto')
    ax3.set_xlabel("Latent Dimensions")
    ax3.set_ylabel("Features")
    plt.colorbar(im, ax=ax3, fraction=0.046)
    ax3.text(0.5, -0.15, "Compressed representation with uncertainty", 
             ha='center', transform=ax3.transAxes)
    
    # FusionAlpha: Knowledge Graph
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_title("FusionAlpha: Multi-Agent Coordination Graph", fontsize=14, fontweight='bold')
    
    # Create knowledge graph
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    agents = ['Agent1', 'Agent2', 'Agent3']
    items = ['Key1', 'Door1', 'Info1', 'Switch1']
    
    for agent in agents:
        G.add_node(agent, node_type='agent')
    for item in items:
        G.add_node(item, node_type='item')
    
    # Add edges (discoveries and coordination)
    G.add_edge('Agent1', 'Key1', relation='discovered')
    G.add_edge('Agent2', 'Door1', relation='discovered')
    G.add_edge('Agent3', 'Info1', relation='discovered')
    G.add_edge('Agent1', 'Agent2', relation='can_communicate')
    G.add_edge('Agent2', 'Agent3', relation='can_communicate')
    G.add_edge('Key1', 'Door1', relation='unlocks')
    G.add_edge('Info1', 'Switch1', relation='reveals')
    
    # Position nodes
    pos = nx.spring_layout(G, k=2, seed=42)
    
    # Draw nodes
    agent_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'agent']
    item_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'item']
    
    nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_color='lightblue', 
                          node_size=1500, ax=ax4)
    nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color='lightgreen', 
                          node_size=1000, ax=ax4)
    
    # Draw edges with different styles
    discovery_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'discovered']
    comm_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'can_communicate']
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') not in ['discovered', 'can_communicate']]
    
    nx.draw_networkx_edges(G, pos, edgelist=discovery_edges, edge_color='green', 
                          style='solid', width=2, ax=ax4)
    nx.draw_networkx_edges(G, pos, edgelist=comm_edges, edge_color='blue', 
                          style='dashed', width=2, ax=ax4)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='red', 
                          style='dotted', width=2, ax=ax4)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax4)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', linewidth=2, label='Discovered'),
        plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Communication'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Knowledge Link'),
        plt.scatter([], [], c='lightblue', s=150, label='Agent'),
        plt.scatter([], [], c='lightgreen', s=100, label='Item')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    ax4.axis('off')
    ax4.text(0.5, -0.05, "Coordinates agents through shared knowledge", 
             ha='center', transform=ax4.transAxes)
    
    # Add main title
    fig.suptitle("BICEP + ENN + FusionAlpha: Triple Pipeline Architecture", 
                 fontsize=16, fontweight='bold')
    
    # Add description
    fig.text(0.5, 0.02, 
             "BICEP handles uncertainty in time → ENN handles uncertainty in state → FusionAlpha handles uncertainty in structure",
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig('triple_pipeline_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("TRIPLE PIPELINE SUMMARY")
    print("="*70)
    print("\n1. BICEP (Brownian Inference):")
    print("   - Generates stochastic exploration paths")
    print("   - Handles temporal uncertainty through Brownian motion")
    print("   - Provides robust path planning in uncertain environments")
    
    print("\n2. ENN (Entangled Neural Network):")
    print("   - Compresses multiple possible states into latent representation")
    print("   - Maintains uncertainty through ensemble methods")
    print("   - Enables efficient processing of high-dimensional observations")
    
    print("\n3. FusionAlpha (Graph-based Coordination):")
    print("   - Builds knowledge graph from agent discoveries")
    print("   - Coordinates multi-agent strategies")
    print("   - Resolves contradictions through graph traversal")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Each component handles a different type of uncertainty:")
    print("- BICEP: Uncertainty in TIME (when/how to move)")
    print("- ENN: Uncertainty in STATE (what is the current situation)")
    print("- FusionAlpha: Uncertainty in STRUCTURE (how things relate)")
    print("="*70)

if __name__ == "__main__":
    demonstrate_triple_pipeline()