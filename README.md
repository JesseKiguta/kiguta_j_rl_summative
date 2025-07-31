
# Urban Flood Management with RL

This project simulates flood management in urban environments using reinforcement learning. It models the dynamics of flooding during heavy rainfall and evaluates strategies for deploying pumps and barriers. The project explores the trade-offs between energy consumption, budget constraints, and water mitigation effectiveness.

Reinforcement learning agents—including **Deep Q-Networks (DQN)** and policy gradient methods like **REINFORCE**, **Proximal Policy Optimization (PPO)**, and **Actor-Critic**—are trained and compared in this custom environment built with Gymnasium and Pygame.

## Problem Statement

Urban areas are increasingly vulnerable to flooding due to climate change and infrastructure challenges. Efficient flood management strategies can prevent disasters, save energy, and reduce costs. This simulation framework enables experimentation with intelligent agents to learn optimal water mitigation strategies under budget and energy constraints.

---

## Project Structure

```
project_root/

├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization GUI components
│   ├── flood_management_random.gif  # GIF of random actions being taken in the environment
│
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/other PG using SB3
│   ├── training_progress.png    # Graph for training performance of the DQN
│   ├── pg_comparison.png        # Graph comparing training performance of the PG methods
│
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
│
├── main.py                      # Entry point for running experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## Environment Overview

- **Grid:** 5×5 city layout (25 cells)
- **Flood Source:** Heavy rainfall causes water accumulation in each cell
- **Mitigation Tools:**
  - **Pumps**: Moderate water reduction, lower energy and budget cost
  - **Barriers**: Heavy water reduction, higher energy and budget cost
- **Goal:** Bring all grid cells to a "safe" water level (green)

###  Action Space

The agent can take the following discrete actions:

| Action             | Description                                  | Cost         |
|--------------------|----------------------------------------------|--------------|
| Place Pump         | Places a pump in a moderately flooded cell   | Low budget, low energy |
| Place Barrier      | Places a barrier in a severely flooded cell  | High budget, high energy |
| No Action          | Chooses to skip intervention on a cell       | No cost      |

Each action is applied to a specific cell during the step.

### State Space

The state representation includes:

- Water level of each cell (normalized float or color-coded: green/orange/red)
- Pump/Barrier presence map (binary for each type)
- Remaining energy and budget

The state is a matrix (e.g., shape `(5,5,3)`) encoding water levels, pump/barrier states, and environmental constraints.

### Reward Function

The reward is calculated based on the effectiveness of interventions:

- **Positive reward** for reducing water level in a cell
- **Negative reward** for:
  - Exceeding budget or energy limits
  - Taking unnecessary actions (e.g., pump on dry cell)
- **Bonus reward** for fully drying a cell or grid

This encourages the agent to optimize water management using the least resources.

---

## Reinforcement Learning Algorithms

Training is done using [Stable Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3):

- **DQN:** Value-based learning with experience replay
- **PPO:** Policy-based learning with clipped surrogate objective
- **REINFORCE / Actor-Critic:** Policy gradient methods with baseline and advantage estimates

Each agent is trained to reduce flooding while minimizing energy and budget costs.

---

## Visualization

- The simulation renders flood states using **Pygame**
- Each cell displays:
  - Water level (via color gradient)
  - Pump/Barrier icons
  - Water volume number
- Animated **GIFs** of episodes are generated via `imageio`

---

## Evaluation Metrics

- **Cumulative Water Removed**
- **Total Energy and Budget Consumed**
- **Episodes to Convergence**
- **Effectiveness Score** = Water Removed / (Budget + Energy)

---

## Running Experiments

```bash
# Train a DQN agent
python training/dqn_training.py

# Train PPO/PG agent
python training/pg_training.py

# Run environment with best performing model
python main.py
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Acknowledgments

Inspired by climate resilience and the need for adaptive urban planning under increasing flood risk. This project was developed as part of a research exploration into **energy-aware reinforcement learning for disaster mitigation**.

---

## License

MIT License