import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from environment.custom_env import FloodEnv
from environment.rendering import FloodEnvRenderer

class StabilizedDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(StabilizedDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256)  # Added layer normalization
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(torch.relu(self.ln2(self.fc2(x))))
        return torch.tanh(self.fc3(x)) * 10  # Constrained output range

class ResilientDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Larger replay buffer
        self.gamma = 0.97  # Slightly reduced discount
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0002  # Reduced learning rate
        self.batch_size = 256  # Larger batch size
        self.model = StabilizedDQN(state_size, action_size)
        self.target_model = StabilizedDQN(state_size, action_size)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=self.learning_rate,
                                   weight_decay=1e-5)  # L2 regularization
        self.update_target_model()
        self.last_loss = None
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        
    def replay(self):
        if len(self.memory) < self.batch_size * 3:  # Wait for more samples
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # Double DQN update
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_actions = self.model(next_states).argmax(1)
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).detach()
        target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Huber loss for stability
        loss = nn.SmoothL1Loss()(current_q, target)
        self.last_loss = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)  # Gradient clipping
        self.optimizer.step()
        
        # Adaptive epsilon adjustment
        if self.last_loss and self.last_loss > 1.0:  # If training is unstable
            self.epsilon = min(0.5, self.epsilon + 0.01)  # Boost exploration
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def preprocess_state(state_dict):
    """Enhanced state preprocessing with normalization"""
    water = state_dict['water_levels'].flatten() / 100.0
    budget = np.log1p(state_dict['budget']) / np.log1p(1000)  # Log-scale
    energy = state_dict['energy'] / 100.0
    pumps = state_dict['pumps'].flatten().astype(float)
    barriers = state_dict['barriers'].flatten().astype(float)
    return np.concatenate([water, budget, energy, pumps, barriers])

def train_dqn(episodes=2000, render_every=100, save_model=True):
    env = FloodEnv()
    renderer = FloodEnvRenderer(env)
    
    state_size = (env.grid_size ** 2) * 3 + 2
    action_size = env.action_space.n
    
    agent = ResilientDQNAgent(state_size, action_size)
    episode_rewards = []
    action_distribution = np.zeros(4)  # Track action types
    
    for e in range(episodes):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 200:  # Max steps per episode
            action = agent.act(state)
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)
            
            # Track action distribution
            action_type = info.get('action_type', action // (env.grid_size**2))
            action_distribution[action_type] += 1
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            if len(agent.memory) > agent.batch_size * 3:
                agent.replay()
                
            if e % render_every == 0:
                renderer.render()
                
        episode_rewards.append(total_reward)
        
        # Adaptive logging
        log_str = (f"Ep {e+1}/{episodes}: "
                  f"Reward={total_reward:7.2f} "
                  f"ε={agent.epsilon:.3f} "
                  f"Loss={agent.last_loss or 0:.3f} "
                  f"Actions=[P:{action_distribution[0]/step:.1%} "
                  f"B:{action_distribution[1]/step:.1%} "
                  f"N:{action_distribution[2]/step:.1%} "
                  f"R:{action_distribution[3]/step:.1%}]")
        print(log_str)
        
        # Reset action tracking
        action_distribution.fill(0)
        
        # Update target network
        if e % 100 == 0:
            agent.update_target_model()
            
        # Early stopping with patience
        if len(episode_rewards) > 100:
            last_100_avg = np.mean(episode_rewards[-100:])
            if last_100_avg < 0 and e > 300:  # Only check after 300 episodes
                print(f"Early stopping - no improvement (avg reward: {last_100_avg:.2f})")
                break
    
    # Save and visualize
    if save_model:
        torch.save({
            'model_state': agent.model.state_dict(),
            'rewards': episode_rewards,
            'epsilon': agent.epsilon
        }, "flood_dqn_model.pth")
    
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'))  # Moving avg
    plt.title("Training Progress (Raw + Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("training_progress.png")
    plt.show()
    
    env.close()
    renderer.close()
    return agent

if __name__ == "__main__":
    trained_agent = train_dqn(episodes=2000, render_every=50)

