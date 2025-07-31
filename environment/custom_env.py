import numpy as np
import pygame
import gym
from gym import spaces
import imageio

class FloodEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(FloodEnv, self).__init__()
        
        # Grid setup
        self.grid_size = 5
        self.cell_size = 100
        self.width = self.grid_size * self.cell_size + 200
        self.height = self.grid_size * self.cell_size
        
        # Water thresholds
        self.low_water = 30    # Green
        self.med_water = 60    # Orange 
        self.high_water = 90   # Red
        
        # Action space (type, x, y)
        self.action_space = spaces.Discrete(4 * self.grid_size * self.grid_size)
        
        # Observation space
        self.observation_space = spaces.Dict({
            "water_levels": spaces.Box(low=0, high=100, shape=(self.grid_size, self.grid_size)),
            "budget": spaces.Box(low=0, high=1000, shape=(1,)),
            "energy": spaces.Box(low=0, high=100, shape=(1,)),
            "pumps": spaces.MultiBinary((self.grid_size, self.grid_size)),
            "barriers": spaces.MultiBinary((self.grid_size, self.grid_size))
        })

        # Resources (adjusted for stability)
        self.initial_budget = 800    # Increased from 500
        self.initial_energy = 150    # Increased from 100
        self.pump_cost = 40          # Reduced from 50
        self.barrier_cost = 80       # Reduced from 100
        self.reinforce_cost = 25     # Reduced from 30
        self.pump_energy = 8         # Reduced from 10
        self.barrier_energy = 15     # Reduced from 20
        self.reinforce_energy = 4    # Reduced from 5

        # Effectiveness (tuned)
        self.pump_effectiveness = 25  # Increased from 20
        self.barrier_effectiveness = 45  # Increased from 40
        self.reinforce_boost = 15     # Increased from 10
        
        # Visualization
        pygame.init()
        self.screen = None
        self.colors = {
            'water': [
                (100, 200, 100),   # Green (low)
                (255, 150, 50),    # Orange (medium)
                (255, 50, 50)      # Red (high)
            ],
            'infra': {
                'pump': (50, 150, 255),
                'barrier': (139, 69, 19),
                'reinforced': (255, 215, 0)
            }
        }
        
        self.reset()

    def reset(self):
        # Initialize with controlled flooding (40-70 instead of 30-70)
        self.water_levels = np.random.randint(40, 70, size=(self.grid_size, self.grid_size))
        
        # Resources
        self.budget = self.initial_budget
        self.energy = self.initial_energy
        
        # Infrastructure
        self.pumps = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.barriers = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.reinforced = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Tracking
        self.total_water_removed = 0
        self.score = 0
        self.last_action = None
        
        return self._get_obs()

    def step(self, action):
        action_type, x, y = self._decode_action(action)
        reward = 0
        done = False
        water_removed = 0
        
        # Action effects (with safety checks)
        if action_type == 0:  # Add pump
            if (self._can_place_pump(x, y)):
                self.pumps[x, y] = True
                self.budget -= self.pump_cost
                self.energy -= self.pump_energy
                reward += 2.0  # Reduced from 5.0
                
        elif action_type == 1:  # Add barrier
            if (self._can_place_barrier(x, y)):
                self.barriers[x, y] = True  
                self.budget -= self.barrier_cost
                self.energy -= self.barrier_energy
                reward += 5.0  # Reduced from 10.0
                
        elif action_type == 3:  # Reinforce
            if (self._can_reinforce(x, y)):
                self.reinforced[x, y] = True
                self.budget -= self.reinforce_cost
                self.energy -= self.reinforce_energy
                reward += 3.0  # Reduced from 7.0

        # Infrastructure effects (water removal)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.pumps[i, j]:
                    removal = min(self.pump_effectiveness, self.water_levels[i, j])
                    self.water_levels[i, j] -= removal
                    water_removed += removal
                    
                if self.barriers[i, j]:
                    effectiveness = self.barrier_effectiveness
                    if self.reinforced[i, j]:
                        effectiveness += self.reinforce_boost
                    removal = min(effectiveness, self.water_levels[i, j])
                    self.water_levels[i, j] -= removal
                    water_removed += removal

        # Rainfall (scaled by current performance)
        rain_intensity = 0.5 + (1 - (water_removed / 100))  # 0.5-1.5 range
        self.water_levels = np.clip(
            self.water_levels + np.random.uniform(0, rain_intensity * 3, (self.grid_size, self.grid_size)),
            0, 100
        )

        # Reward calculation (stabilized)
        remaining_water = np.sum(self.water_levels)
        self.total_water_removed += water_removed
        reward += 0.2 * water_removed  # Increased from 0.1
        reward -= 0.5  # Step penalty (reduced from 1.0)
        
        # Terminal conditions
        if np.all(self.water_levels <= self.low_water):
            reward += 20.0  # Reduced from 100
            done = True
        elif np.any(self.water_levels >= 100):
            reward -= 10.0  # Reduced from 50
            done = True
        elif self.budget <= 0 or self.energy <= 0:
            done = True
            
        # Emergency reward stabilization
        reward = max(reward, -10.0)  # Cap minimum reward
        reward = reward / 10.0  # Normalize to [-1, 1] range
        
        self.score = self.total_water_removed - remaining_water * 0.1
        self.last_action = action_type
        
        return self._get_obs(), reward, done, {
            "score": self.score,
            "water_removed": water_removed,
            "action_type": action_type,
            "remaining_water": remaining_water
        }

    def _can_place_pump(self, x, y):
        return (not self.pumps[x, y] and not self.barriers[x, y] 
                and self.water_levels[x, y] >= self.low_water
                and self.budget >= self.pump_cost 
                and self.energy >= self.pump_energy)

    def _can_place_barrier(self, x, y):
        return (not self.barriers[x, y] and not self.pumps[x, y]
                and self.water_levels[x, y] >= self.med_water
                and self.budget >= self.barrier_cost
                and self.energy >= self.barrier_energy)

    def _can_reinforce(self, x, y):
        return (self.barriers[x, y] and not self.reinforced[x, y]
                and self.budget >= self.reinforce_cost
                and self.energy >= self.reinforce_energy)

    def _decode_action(self, action):
        action_type = action // (self.grid_size * self.grid_size)
        remaining = action % (self.grid_size * self.grid_size)
        x = remaining // self.grid_size
        y = remaining % self.grid_size
        return action_type, x, y

    def _get_obs(self):
        return {
            "water_levels": self.water_levels.copy(),
            "budget": np.array([self.budget]),
            "energy": np.array([self.energy]),
            "pumps": self.pumps.copy(),
            "barriers": self.barriers.copy()
        }

    def render(self, mode='human'):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.screen.fill((240, 240, 240))
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                water_level = self.water_levels[i, j]
                if water_level < self.low_water:
                    color_idx = 0
                elif water_level < self.med_water:
                    color_idx = 1
                else:
                    color_idx = 2
                
                pygame.draw.rect(
                    self.screen, self.colors['water'][color_idx],
                    (j*self.cell_size + 2, i*self.cell_size + 2,
                     self.cell_size - 4, self.cell_size - 4),
                    border_radius=5
                )
                
                # Draw infrastructure
                if self.pumps[i, j]:
                    pygame.draw.circle(
                        self.screen, self.colors['infra']['pump'],
                        (j*self.cell_size + self.cell_size//2,
                         i*self.cell_size + self.cell_size//2),
                        15
                    )
                elif self.barriers[i, j]:
                    color = self.colors['infra']['reinforced'] if self.reinforced[i, j] else self.colors['infra']['barrier']
                    pygame.draw.rect(
                        self.screen, color,
                        (j*self.cell_size + self.cell_size//2 - 15,
                         i*self.cell_size + self.cell_size//2 - 15, 30, 30)
                    )
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

def create_random_agent_gif(env, filename='flood_management_random.gif', max_frames=100):
    frames = []
    obs = env.reset()
    done = False
    frame_count = 0
    
    while not done and frame_count < max_frames:
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Render and add to frames
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        frame_count += 1
    
    # Save as GIF
    with imageio.get_writer(filename, fps=5) as writer:
        for frame in frames:
            writer.append_data(frame)
    
    return filename

# Create and test the environment
if __name__ == "__main__":
    env = FloodEnv()
    gif_path = create_random_agent_gif(env)
    print(f"GIF saved at: {gif_path}")

