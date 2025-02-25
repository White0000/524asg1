# 524asg1
5571004

# RoboCleaner: A Reinforcement Learning Game

This repository hosts a complete Reinforcement Learning (RL) project called **RoboCleaner**, featuring multiple UIs (PyGame, Tkinter, PyQt), multiple RL algorithms (Q-learning, SARSA, DQN, PPO), and randomly generated or preset levels. The robot cleans dirt in a grid-like environment with obstacles, energy constraints, or time limits. This project demonstrates how to integrate RL agents, environment design, and user interfaces in Python.

------

## Overview

RoboCleaner spawns an agent in a grid-based world with obstacles and dirt cells. The agent can move up, down, left, right, or perform a special action (e.g., cleaning). Various RL algorithms guide the agent toward maximizing rewards. Users can run the game with different algorithms and choose from multiple GUIs or even a random map generation mode. The difficulty can scale automatically to apply energy/time constraints or increase penalty/reward values.

------

## Features

- Multiple RL Algorithms:
  - Q-Learning
  - SARSA
  - DQN (with options like dueling, prioritized replay, double-DQN)
  - PPO
- Multiple GUIs:
  - **PyGame** for a 2D animated experience
  - **Tkinter** for a simpler windowed UI
  - **PyQt** for a more modern interface
- Random or Preset Levels:
  - 5 automatically generated levels with adjustable difficulty
  - You can create or load custom levels from JSON files
- Difficulty-based Constraints:
  - Lower difficulty (1–2): More energy, larger time limits
  - Medium difficulty (3): Balanced energy and time
  - Higher difficulty (4–5): Strict time limits, lower energy, heavier penalties
- Keyboard Input (Optional):
  - Agents can be purely RL-driven or partially controlled with arrow keys (if integrated in the environment step).
- Extendable:
  - You can add traps, teleports, custom tile types, or multi-agent collaboration/adversarial modes.

------

## Project Structure

```
gamellast/
  ├── environment/
  │    └── environment.py        # The RoboCleanerEnv environment
  ├── agents/
  │    ├── qlearning.py
  │    ├── sarsa.py
  │    ├── dqn.py
  │    └── ppo.py
  ├── ui/
  │    ├── pygame_ui.py
  │    ├── tkinter_ui.py
  │    └── pyqt_ui.py
  ├── levels/
  │    └── levels.py             # LEVELS array, random generation, etc.
  ├── utils/
  │    └── utils.py              # Utility methods for obstacles/dirt
  ├── main.py                    # Multi-UI entry script with optional dialogs
  ├── run_game.py                # Core loop with chosen agent & environment
  ├── README.md                  # (this file)
  └── ... (additional config or assets)
```

- **environment.py** – Defines the `RoboCleanerEnv` environment logic.
- **levels.py** – Manages level data, random generation, difficulty scaling.
- **agents** – Contains Q-learning, SARSA, DQN, PPO implementations.
- **ui** – Various user interfaces for rendering and interacting with the environment.
- **main.py** – Extended script for launching single or multiple UIs concurrently.
- **run_game.py** – Standard RL loop that binds an agent with a UI and environment.

------

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YourName/RoboCleaner.git
   ```

2. Navigate to the project folder:

   ```bash
   cd RoboCleaner
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   (If `requirements.txt` is provided; otherwise manually install packages: `pygame`, `PyQt5`, etc.)

4. (Optional) To run the PyQt or Tkinter UI, confirm you have those libraries installed:

   - **Tkinter** typically ships with standard Python on many systems.
   - **PyQt** can be installed via `pip install pyqt5`.

------

## Usage

### 1. Single UI Launch

You can launch the game with a single UI:

```bash
python main.py --ui pygame
```

or

```bash
python main.py --ui tkinter --algo dqn --episodes 10 --fps 5
```

When multiple UIs are not specified, it defaults to a single UI process.

### 2. Multi-UI Launch (dialog-driven)

By default, `main.py` can present a dialog to select multiple UIs. You can skip dialogs or pass config overrides.

### 3. run_game.py Direct Launch

If you prefer direct CLI usage:

```bash
python run_game.py --ui pygame --algo sarsa --episodes 3
```

------

## Level System

- `levels.py` uses a global array `LEVELS`. Each level has rows, cols, obstacles, dirts, difficulty, energy, time_limit, etc.

- `build_random_five_levels()` automatically creates 5 random levels with increasing difficulty.

- Difficulty

   influences 

  ```
  energy
  ```

   and 

  ```
  time_limit
  ```

  - `1–2`: More generous resources
  - `3` : Balanced
  - `4–5`: Stricter resources

------

## Agents

- **QLearningAgent** – Basic Q-table approach
- **SarsaAgent** – On-policy temporal-difference
- **DQNAgent** – Deep Q-learning with optional dueling, double-DQN, prioritized replay
- **PPOAgent** – Policy gradient with advantage estimation

Agents can be configured via command-line or code overrides (alpha, epsilon, etc.). They implement standard RL loops, storing transitions and learning from them.

------

## User Interfaces

- **PyGame**: Animated 2D, tile-based display with optional smooth movement for the agent sprite.
- **Tkinter**: Simpler windowed interface, static rendering.
- **PyQt**: Another alternative for GUI, more modern widgets.

Each UI:

- Renders the environment grid (obstacles, dirt, agent).
- Displays side stats (score, level, difficulty, losses, etc.).
- Provides a menu or button bar for toggling metrics, speed, training loops, etc.

------

## Customization

- Modify or add new levels in `levels.py`.
- Tweak environment logic in `environment.py` (e.g., add traps, teleports).
- Adjust agent hyperparameters (learning rate, exploration) in `agents/`.
- Switch from random_map to a preset or loaded map via CLI flags (`--random_map`).
- Extend UI code in `pygame_ui.py`, `tkinter_ui.py`, or `pyqt_ui.py` to add custom buttons or visual effects.

------

## License

MIT License

------

Enjoy training your RL agent in **RoboCleaner**! 

If you encounter issues, feel free to open an issue or contribute improvements.

