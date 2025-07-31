import torch
import pygame
import numpy as np
from environment.custom_env import FloodEnv
from environment.rendering import FloodEnvRenderer
import time


def load_actor_critic(model_path, state_size, action_size):
    """Safely load trained Actor-Critic model"""
    from training.pg_training import ActorCritic
    
    # Allow necessary globals for numpy compatibility
    allowed_globals = [np.core.multiarray.scalar]
    with torch.serialization.safe_globals(allowed_globals):
        checkpoint = torch.load(model_path, weights_only=False)
    
    model = ActorCritic(state_size, action_size)
    model.policy.load_state_dict(checkpoint['model_state'])
    model.value.load_state_dict(checkpoint['value_state'])
    
    print(f"Successfully loaded Actor-Critic model from {model_path}")
    return model

def preprocess_state(state_dict):
    """Consistent state preprocessing"""
    water = state_dict['water_levels'].flatten() / 100.0
    budget = state_dict['budget'] / 1000.0
    energy = state_dict['energy'] / 100.0
    pumps = state_dict['pumps'].flatten().astype(float)
    barriers = state_dict['barriers'].flatten().astype(float)
    return np.concatenate([water, budget, energy, pumps, barriers])

def run_episode(env, renderer, model, max_steps=1000, speed=0.1):
    """Run visualization episode"""
    state = preprocess_state(env.reset())
    total_reward = 0
    
    for step in range(max_steps):
        with torch.no_grad():
            action = model.select_action(state)
        
        next_state_dict, reward, done, _ = env.step(action)
        state = preprocess_state(next_state_dict)
        total_reward += reward
        
        renderer.render()
        time.sleep(speed)
        
        if done or any(event.type == pygame.QUIT for event in pygame.event.get()):
            break
    
    print(f"Episode completed in {step+1} steps | Reward: {total_reward:.1f}")
    return total_reward

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/pg/ActorCritic/ActorCritic_ep_500.pth"  # Your saved model
    RENDER_SPEED = 0.2  # Visualization speed (seconds/frame)
    EPISODES = 3  # Number of episodes to run
    
    # Initialize
    env = FloodEnv()
    renderer = FloodEnvRenderer(env)
    
    try:
        # Load model safely
        state_size = (env.grid_size ** 2) * 3 + 2
        action_size = env.action_space.n
        model = load_actor_critic(MODEL_PATH, state_size, action_size)
        
        # Run episodes
        for ep in range(1, EPISODES+1):
            print(f"\n=== Episode {ep}/{EPISODES} ===")
            run_episode(env, renderer, model, speed=RENDER_SPEED)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        env.close()
        renderer.close()
        pygame.quit()

