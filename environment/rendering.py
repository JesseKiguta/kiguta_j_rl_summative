import pygame
import numpy as np
from environment.custom_env import FloodEnv  # Import custom environment

class FloodEnvRenderer:
    def __init__(self, env):
        self.env = env
        pygame.init()
        
        # Screen setup
        self.screen_width = env.width
        self.screen_height = env.height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flood Management Simulation")
        
        # Fonts
        self.font_large = pygame.font.SysFont('Arial', 24)
        self.font_medium = pygame.font.SysFont('Arial', 20)
        self.font_small = pygame.font.SysFont('Arial', 16)
        
        # Colors (expanded palette)
        self.colors = {
            'water_green': (100, 200, 100),
            'water_orange': (255, 150, 50),
            'water_red': (255, 50, 50),
            'pump_blue': (50, 150, 255),
            'barrier_brown': (139, 69, 19),
            'reinforced_gold': (255, 215, 0),
            'text_dark': (50, 50, 50),
            'background': (240, 240, 240),
            'grid_line': (200, 200, 200)
        }
        
        # Load or create icons
        self._create_icons()
        
        # Animation control
        self.clock = pygame.time.Clock()
        self.fps = 5
    
    def _create_icons(self):
        """Create graphical elements for rendering"""
        # Water level indicators
        self.water_icons = {
            'low': self._create_water_icon(self.colors['water_green']),
            'medium': self._create_water_icon(self.colors['water_orange']),
            'high': self._create_water_icon(self.colors['water_red'])
        }
        
        # Infrastructure icons
        self.infrastructure_icons = {
            'pump': self._create_pump_icon(),
            'barrier': self._create_barrier_icon(),
            'reinforced': self._create_reinforced_icon()
        }
    
    def _create_water_icon(self, color):
        """Create a water level indicator icon"""
        icon = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(icon, color, (0, 0, 30, 30), border_radius=5)
        return icon
    
    def _create_pump_icon(self):
        """Create a pump icon"""
        icon = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(icon, self.colors['pump_blue'], (15, 15), 12)
        # Add pump details
        pygame.draw.rect(icon, (255, 255, 255), (12, 10, 6, 10))
        return icon
    
    def _create_barrier_icon(self):
        """Create a barrier icon"""
        icon = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(icon, self.colors['barrier_brown'], (5, 5, 20, 20))
        return icon
    
    def _create_reinforced_icon(self):
        """Create a reinforced barrier icon"""
        icon = self._create_barrier_icon()
        pygame.draw.line(icon, self.colors['reinforced_gold'], (5, 5), (25, 25), 3)
        pygame.draw.line(icon, self.colors['reinforced_gold'], (25, 5), (5, 25), 3)
        return icon
    
    def render(self):
        """Main rendering function - now only supports real-time display"""
        self.screen.fill(self.colors['background'])
        
        # Draw the grid and cells
        self._render_grid()
        
        # Draw the legend/info panel
        self._render_legend()
        
        # Draw the infrastructure
        self._render_infrastructure()
        
        # Draw the water levels
        self._render_water_levels()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _render_grid(self):
        """Draw the grid lines"""
        for i in range(self.env.grid_size + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen, self.colors['grid_line'],
                (i * self.env.cell_size, 0),
                (i * self.env.cell_size, self.env.grid_size * self.env.cell_size),
                2
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen, self.colors['grid_line'],
                (0, i * self.env.cell_size),
                (self.env.grid_size * self.env.cell_size, i * self.env.cell_size),
                2
            )
    
    def _render_water_levels(self):
        """Draw the water levels in each cell"""
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                water_level = self.env.water_levels[i, j]
                
                # Determine water color
                if water_level < self.env.low_water:
                    color = self.colors['water_green']
                elif water_level < self.env.med_water:
                    color = self.colors['water_orange']
                else:
                    color = self.colors['water_red']
                
                # Draw water level background
                rect = pygame.Rect(
                    j * self.env.cell_size + 2,
                    i * self.env.cell_size + 2,
                    self.env.cell_size - 4,
                    self.env.cell_size - 4
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                # Display water level number
                text = self.font_small.render(f"{int(water_level)}", True, self.colors['text_dark'])
                self.screen.blit(text, (j * self.env.cell_size + 10, i * self.env.cell_size + 10))
    
    def _render_infrastructure(self):
        """Draw pumps and barriers"""
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                center_x = j * self.env.cell_size + self.env.cell_size // 2
                center_y = i * self.env.cell_size + self.env.cell_size // 2
                
                if self.env.pumps[i, j]:
                    icon = self.infrastructure_icons['pump']
                    self.screen.blit(icon, (center_x - 15, center_y - 15))
                elif self.env.barriers[i, j]:
                    if self.env.reinforced[i, j]:
                        icon = self.infrastructure_icons['reinforced']
                    else:
                        icon = self.infrastructure_icons['barrier']
                    self.screen.blit(icon, (center_x - 15, center_y - 15))
    
    def _render_legend(self):
        """Draw the information panel on the right"""
        panel_x = self.env.grid_size * self.env.cell_size + 10
        panel_width = self.screen_width - panel_x - 10
        
        # Draw panel background
        pygame.draw.rect(
            self.screen, (220, 220, 220),
            (panel_x, 10, panel_width, self.screen_height - 20),
            border_radius=10
        )
        
        # Water level legend
        title = self.font_medium.render("Water Levels", True, self.colors['text_dark'])
        self.screen.blit(title, (panel_x + 10, 20))
        
        y_offset = 50
        for level, color_key in [("Low", 'water_green'), 
                               ("Medium", 'water_orange'), 
                               ("High", 'water_red')]:
            pygame.draw.rect(
                self.screen, self.colors[color_key],
                (panel_x + 15, y_offset, 25, 25), border_radius=5
            )
            text = self.font_small.render(level, True, self.colors['text_dark'])
            self.screen.blit(text, (panel_x + 50, y_offset + 5))
            y_offset += 35
        
        # Infrastructure legend
        y_offset += 20
        title = self.font_medium.render("Infrastructure", True, self.colors['text_dark'])
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 30
        
        for infra, label in [("pump", "Pump"), 
                           ("barrier", "Barrier"), 
                           ("reinforced", "Reinforced")]:
            self.screen.blit(
                self.infrastructure_icons[infra],
                (panel_x + 15, y_offset)
            )
            text = self.font_small.render(label, True, self.colors['text_dark'])
            self.screen.blit(text, (panel_x + 50, y_offset + 5))
            y_offset += 35
        
        # Resources and score
        y_offset += 20
        title = self.font_medium.render("Resources", True, self.colors['text_dark'])
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 30
        
        for label, value in [("Budget", f"${int(self.env.budget)}"), 
                           ("Energy", f"{int(self.env.energy)} units"),
                           ("Score", f"{int(self.env.score)}")]:
            text = self.font_small.render(f"{label}: {value}", True, self.colors['text_dark'])
            self.screen.blit(text, (panel_x + 15, y_offset))
            y_offset += 25
    
    def close(self):
        pygame.quit()


# Example usage for testing just the renderer
if __name__ == "__main__":
    env = FloodEnv()
    renderer = FloodEnvRenderer(env)
    
    # Test rendering
    obs = env.reset()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Render only
        renderer.render()
        
        if done:
            obs = env.reset()
    
    renderer.close()

