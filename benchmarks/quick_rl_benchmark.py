#!/usr/bin/env python3
"""
Quick RL Benchmark: BICEP+ENN vs Baselines on Navigation
Focused benchmark for faster results
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from collections import deque
import matplotlib.pyplot as plt

class SimpleGridWorld:
    """Simplified grid world environment"""
    
    def __init__(self, size=10):
        self.size = size
        self.reset()
        
    def reset(self):
        # Simple grid with some obstacles
        self.grid = np.zeros((self.size, self.size))
        
        # Add a few obstacles
        obstacles = [(3, 3), (3, 4), (6, 6), (6, 7), (7, 6)]
        for x, y in obstacles:
            if x < self.size and y < self.size:
                self.grid[x, y] = -1
                
        self.agent_pos = (1, 1)
        self.goal_pos = (8, 8)
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self):
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        
        state = [
            x / self.size, y / self.size,  # Agent position
            gx / self.size, gy / self.size,  # Goal position
            (gx - x) / self.size, (gy - y) / self.size,  # Direction to goal
            np.sqrt((gx - x)**2 + (gy - y)**2) / (self.size * np.sqrt(2))  # Distance to goal
        ]
        
        # Add obstacle info around agent
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    state.append(1.0 if self.grid[nx, ny] == -1 else 0.0)
                else:
                    state.append(1.0)  # Boundary
                    
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        
        # Check if valid move (not into obstacle)
        if self.grid[new_x, new_y] != -1:
            self.agent_pos = (new_x, new_y)
            
        self.steps += 1
        
        # Reward: -0.1 for each step, +10 for reaching goal
        reward = -0.1
        done = False
        
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        elif self.steps >= 50:  # Max steps
            done = True
            
        return self._get_state(), reward, done, {}


class SimpleQ(nn.Module):
    """Simple Q-network"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class EnsembleQ(nn.Module):
    """ENN-style ensemble Q-network"""
    def __init__(self, state_dim, action_dim, n_heads=3):
        super().__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multiple heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            ) for _ in range(n_heads)
        ])
        
    def forward(self, x, return_all=False):
        features = self.features(x)
        outputs = [head(features) for head in self.heads]
        
        if return_all:
            return outputs
        else:
            return torch.stack(outputs).mean(0)


class DQNAgent:
    """Standard DQN"""
    def __init__(self, state_dim, action_dim):
        self.q_net = SimpleQ(state_dim, action_dim)
        self.target_net = SimpleQ(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.95
        
        self.update_target()
        
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(4)
        
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q_vals.argmax().item()
            
    def train_step(self):
        if len(self.memory) < 32:
            return
            
        batch = np.random.choice(len(self.memory), 32, replace=False)
        states = torch.FloatTensor(np.array([self.memory[i][0] for i in batch]))
        actions = torch.LongTensor(np.array([self.memory[i][1] for i in batch]))
        rewards = torch.FloatTensor(np.array([self.memory[i][2] for i in batch]))
        next_states = torch.FloatTensor(np.array([self.memory[i][3] for i in batch]))
        dones = torch.FloatTensor(np.array([self.memory[i][4] for i in batch]))
        
        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_vals = self.target_net(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_vals
            
        loss = F.mse_loss(q_vals, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class BICEPENNAgent:
    """BICEP+ENN Agent"""
    def __init__(self, state_dim, action_dim):
        self.q_net = EnsembleQ(state_dim, action_dim, n_heads=3)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        
        self.memory = deque(maxlen=1000)
        self.demo_memory = deque(maxlen=500)
        
        self.temperature = 2.0
        self.temp_decay = 0.99
        self.min_temp = 0.3
        self.gamma = 0.95
        
    def act(self, state, training=True):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if training:
                # Get all head outputs for uncertainty estimation
                head_outputs = self.q_net(state_tensor, return_all=True)
                q_vals = torch.stack(head_outputs)
                
                # Use uncertainty for exploration
                q_mean = q_vals.mean(0)
                q_std = q_vals.std(0)
                
                # BICEP-style exploration with uncertainty bonus
                exploration_bonus = q_std * 0.5
                q_explore = q_mean + exploration_bonus
                
                # Temperature-based selection
                probs = F.softmax(q_explore / self.temperature, dim=-1)
                action = torch.distributions.Categorical(probs).sample().item()
            else:
                q_vals = self.q_net(state_tensor)
                action = q_vals.argmax().item()
                
        return action
        
    def generate_demo(self, env, n_episodes=5):
        """Generate BICEP-style demonstrations"""
        demonstrations = []
        
        for _ in range(n_episodes):
            state = env.reset()
            episode = []
            done = False
            
            while not done:
                # BICEP: try multiple actions, pick best based on value + noise
                best_action = 0
                best_value = -float('inf')
                
                for action in range(4):
                    with torch.no_grad():
                        q_vals = self.q_net(torch.FloatTensor(state).unsqueeze(0))
                        value = q_vals[0, action].item() + np.random.normal(0, 0.5)
                        
                    if value > best_value:
                        best_value = value
                        best_action = action
                        
                next_state, reward, done, _ = env.step(best_action)
                episode.append((state, best_action, reward, next_state, done))
                state = next_state
                
            demonstrations.extend(episode)
            
        return demonstrations
        
    def train_on_demos(self, demonstrations):
        """Train on BICEP demonstrations"""
        if not demonstrations:
            return
            
        for demo in demonstrations:
            state, action, reward, next_state, done = demo
            
            with torch.no_grad():
                next_q = self.q_net(torch.FloatTensor(next_state).unsqueeze(0)).max().item()
                target = reward + (1 - done) * self.gamma * next_q
                
            q_pred = self.q_net(torch.FloatTensor(state).unsqueeze(0))[0, action]
            loss = F.mse_loss(q_pred, torch.FloatTensor([target]))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def train_step(self):
        if len(self.memory) < 32:
            return
            
        batch = np.random.choice(len(self.memory), 32, replace=False)
        states = torch.FloatTensor(np.array([self.memory[i][0] for i in batch]))
        actions = torch.LongTensor(np.array([self.memory[i][1] for i in batch]))
        rewards = torch.FloatTensor(np.array([self.memory[i][2] for i in batch]))
        next_states = torch.FloatTensor(np.array([self.memory[i][3] for i in batch]))
        dones = torch.FloatTensor(np.array([self.memory[i][4] for i in batch]))
        
        # Ensemble Q-learning
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.q_net(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q
            
        loss = F.mse_loss(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay temperature
        if self.temperature > self.min_temp:
            self.temperature *= self.temp_decay


def train_agent(agent_class, agent_name, n_episodes=100):
    """Train and evaluate agent"""
    env = SimpleGridWorld()
    state_dim = len(env._get_state())
    agent = agent_class(state_dim, 4)
    
    print(f"\nTraining {agent_name}...")
    
    # Pre-training for BICEP+ENN
    if isinstance(agent, BICEPENNAgent):
        print("  Generating BICEP demonstrations...")
        demos = agent.generate_demo(env, n_episodes=10)
        agent.train_on_demos(demos)
        
    # Main training
    rewards = []
    success_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
        rewards.append(total_reward)
        if env.agent_pos == env.goal_pos:
            success_count += 1
            
        # Update target network for DQN
        if isinstance(agent, DQNAgent) and episode % 10 == 0:
            agent.update_target()
            
        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(rewards[-25:])
            success_rate = success_count / (episode + 1)
            print(f"  Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Success Rate = {success_rate:.1%}")
            
    # Evaluation
    print(f"  Evaluating {agent_name}...")
    eval_rewards = []
    eval_success = 0
    
    for _ in range(20):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
        eval_rewards.append(total_reward)
        if env.agent_pos == env.goal_pos:
            eval_success += 1
            
    return {
        'training_rewards': rewards,
        'eval_reward_mean': np.mean(eval_rewards),
        'eval_reward_std': np.std(eval_rewards),
        'eval_success_rate': eval_success / 20,
        'final_train_reward': np.mean(rewards[-10:])
    }


def main():
    print("=" * 60)
    print("QUICK BICEP+ENN vs DQN NAVIGATION BENCHMARK")
    print("=" * 60)
    
    # Test both agents
    agents = [
        (DQNAgent, "DQN"),
        (BICEPENNAgent, "BICEP+ENN")
    ]
    
    results = {}
    
    for agent_class, agent_name in agents:
        start_time = time.time()
        result = train_agent(agent_class, agent_name, n_episodes=100)
        result['train_time'] = time.time() - start_time
        results[agent_name] = result
        
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<15} {'Train Reward':<12} {'Eval Reward':<15} {'Success Rate':<12} {'Time (s)'}")
    print("-" * 60)
    
    for agent_name, result in results.items():
        print(f"{agent_name:<15} {result['final_train_reward']:<12.2f} "
              f"{result['eval_reward_mean']:<8.2f}±{result['eval_reward_std']:<5.2f} "
              f"{result['eval_success_rate']:<12.1%} {result['train_time']:<8.1f}")
    
    # Comparison
    if len(results) == 2:
        dqn_success = results['DQN']['eval_success_rate']
        bicep_success = results['BICEP+ENN']['eval_success_rate']
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"BICEP+ENN vs DQN Success Rate: {bicep_success/max(dqn_success, 0.01):.2f}x")
        
        if bicep_success > dqn_success:
            print("🎉 BICEP+ENN outperforms DQN!")
        elif bicep_success > dqn_success * 0.9:
            print("✅ BICEP+ENN performs competitively with DQN")
        else:
            print("⚠️  DQN outperforms BICEP+ENN in this task")
            
        print("\nKey Insights:")
        print("• BICEP provides stochastic exploration strategies")
        print("• ENN ensemble reduces overfitting and estimates uncertainty")
        print("• Temperature-based action selection balances exploration/exploitation")
        print("• Pre-training on demonstrations gives head start")


if __name__ == "__main__":
    main()