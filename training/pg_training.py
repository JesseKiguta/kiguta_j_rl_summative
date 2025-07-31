import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from environment.custom_env import FloodEnv
import os
import time
from torch.distributions import Categorical

# Shared Network Architecture
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return torch.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

# REINFORCE Algorithm
class REINFORCE:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0005)
        self.gamma = 0.99
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self):
        # Skip update if no actions were taken
        if len(self.rewards) == 0 or len(self.saved_log_probs) == 0:
            print("Warning: Empty episode - skipping update")
            self.rewards.clear()
            self.saved_log_probs.clear()
            return
            
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Verify matching lengths
        if len(returns) != len(self.saved_log_probs):
            min_length = min(len(returns), len(self.saved_log_probs))
            returns = returns[:min_length]
            self.saved_log_probs = self.saved_log_probs[:min_length]
            print(f"Warning: Length mismatch - truncated to {min_length}")
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Only proceed if we have valid losses
        if len(policy_loss) > 0:
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()  # Changed from cat to stack
            policy_loss.backward()
            self.optimizer.step()
        
        self.rewards.clear()
        self.saved_log_probs.clear()

# Actor-Critic Algorithm
class ActorCritic:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.value = ValueNetwork(state_size)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + 
                                  list(self.value.parameters()), lr=0.0003)
        self.gamma = 0.99
        self.saved_log_probs = []
        self.rewards = []
        self.state_values = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        state_value = self.value(state)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        
        self.saved_log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        return action.item()
    
    def update_policy(self):
        # Skip update if no data collected
        if len(self.rewards) == 0:
            print("Warning: Empty episode - skipping update")
            self.rewards.clear()
            self.saved_log_probs.clear()
            self.state_values.clear()
            return
            
        R = 0
        policy_loss = []
        value_loss = []
        returns = []
        
        # Calculate returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        state_values = torch.cat(self.state_values)
        advantages = returns - state_values.detach()
        
        # Calculate policy loss
        if len(self.saved_log_probs) != len(advantages):
            min_length = min(len(self.saved_log_probs), len(advantages))
            self.saved_log_probs = self.saved_log_probs[:min_length]
            advantages = advantages[:min_length]
            print(f"Warning: Length mismatch - truncated to {min_length}")
        
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
            
        # Calculate value loss
        value_loss = nn.MSELoss()(state_values.squeeze(), returns)
        
        # Only update if we have valid losses
        if len(policy_loss) > 0:
            self.optimizer.zero_grad()
            total_loss = torch.stack(policy_loss).sum() + value_loss  # Changed from cat to stack
            total_loss.backward()
            self.optimizer.step()
        
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.state_values.clear()

# PPO Algorithm
class PPO:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.value = ValueNetwork(state_size)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + 
                                      list(self.value.parameters()), lr=0.0002)
        self.gamma = 0.99
        self.eps_clip = 0.1
        self.K_epochs = 8
        self.mini_batch_size = 64  # Add this line
        self.buffer = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.policy(state)
            state_value = self.value(state)
        m = Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action), state_value
    
    def update_policy(self):
        # Skip update if buffer is empty
        if len(self.buffer) == 0:
            print("Warning: Empty buffer - skipping update")
            return
            
        # Convert buffer to tensors
        old_states = torch.FloatTensor(np.array([t[0] for t in self.buffer]))
        old_actions = torch.LongTensor(np.array([t[1] for t in self.buffer]))
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in self.buffer]))
        rewards = torch.FloatTensor(np.array([t[4] for t in self.buffer]))
        dones = torch.FloatTensor(np.array([t[5] for t in self.buffer]))

        # Calculate returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # Normalize advantages
        with torch.no_grad():
            old_values = self.value(old_states)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Create random indices for mini-batches
            indices = np.arange(len(old_states))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                # Get mini-batch
                mb_states = old_states[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Evaluate actions
                new_probs = self.policy(mb_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(mb_actions).unsqueeze(1)
                new_values = self.value(mb_states)
                entropy = dist.entropy().mean()

                # Policy loss
                ratios = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratios * mb_advantages.detach()
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * mb_advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_values, mb_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear buffer after update
        self.buffer = []

def preprocess_state(state_dict):
    """Shared state preprocessing"""
    water = state_dict['water_levels'].flatten() / 100.0
    budget = state_dict['budget'] / 1000.0
    energy = state_dict['energy'] / 100.0
    pumps = state_dict['pumps'].flatten().astype(float)
    barriers = state_dict['barriers'].flatten().astype(float)
    return np.concatenate([water, budget, energy, pumps, barriers])

def train_pg_algorithm(algorithm, episodes=1000, save_interval=50, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    env = FloodEnv()
    state_size = (env.grid_size ** 2) * 3 + 2
    action_size = env.grid_size ** 2 * 4  # All possible actions
    
    agent = algorithm(state_size, action_size)
    rewards = []
    algo_name = algorithm.__name__
    
    # Create subdirectory for this algorithm
    algo_dir = os.path.join(save_dir, algo_name)
    os.makedirs(algo_dir, exist_ok=True)
    
    for e in range(episodes):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False
        
        while not done:
            if isinstance(agent, PPO):
                action, log_prob, state_value = agent.select_action(state)
                next_state_dict, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state_dict)
                agent.buffer.append((state, action, log_prob, state_value, reward, float(done)))
            else:
                action = agent.select_action(state)
                next_state_dict, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state_dict)
                agent.rewards.append(reward)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update policy
        if isinstance(agent, PPO):
            agent.update_policy()
        else:
            agent.update_policy()
        
        rewards.append(total_reward)
        
        # Save model at intervals
        if (e + 1) % save_interval == 0 or (e + 1) == episodes:
            checkpoint = {
                'episode': e + 1,
                'model_state': agent.policy.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'reward': total_reward,
                'avg_reward': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
                'all_rewards': rewards
            }
            
            if isinstance(agent, (ActorCritic, PPO)):
                checkpoint['value_state'] = agent.value.state_dict()
            
            torch.save(
                checkpoint,
                os.path.join(algo_dir, f"{algo_name}_ep_{e+1}.pth")
            )
            print(f"Saved {algo_name} checkpoint at episode {e+1}")
        
        # Print progress
        avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        print(f"Ep {e+1}/{episodes} | {algo_name} | Reward: {total_reward:.1f} | Avg 50: {avg_reward:.1f}")
        
        # Early stopping
        if len(rewards) > 100 and avg_reward < 0:
            print("Early stopping - no improvement")
            break
    
    return rewards

def compare_algorithms(episodes=1000, save_interval=50, save_dir="models"):
    algorithms = [REINFORCE, ActorCritic, PPO]
    results = {}
    
    for algo in algorithms:
        print(f"\n=== Training {algo.__name__} ===")
        start_time = time.time()
        rewards = train_pg_algorithm(
            algo,
            episodes=episodes,
            save_interval=save_interval,
            save_dir=save_dir
        )
        results[algo.__name__] = {
            'rewards': rewards,
            'time': time.time() - start_time
        }
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(data['rewards'], label=f"{name} (Time: {data['time']:.1f}s)")
    
    plt.title("Policy Gradient Algorithms Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "pg_comparison.png"))
    plt.show()

if __name__ == "__main__":
    compare_algorithms(episodes=500, save_interval=25)

