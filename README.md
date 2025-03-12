# Sailboat Reinforcement Learning Challenge

This project implements reinforcement learning algorithms to train an autonomous sailboat to navigate to a checkpoint in a simulated environment.

## Project Overview

The project simulates a sailboat navigating in a 2D environment with wind forces. The goal is to reach a checkpoint (represented as a red circle) using two controls:
- Sail angle
- Rudder angle

Two reinforcement learning algorithms are implemented:
- Deep Q-Network (DQN)
- Deep Deterministic Policy Gradient (DDPG)

## Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Tkinter (for visualization)

Install the required packages:
```bash
pip install -r requirement.txt
```

## Environment

The sailboat environment is implemented in `environnement/env.py`. It provides:
- Realistic physics simulation of sailboat movement
- Wind forces affecting the sailboat
- Reward system based on reaching the checkpoint

### Environment Parameters

- `batch_size`: Number of environments to run in parallel
- `checkpoint_radius`: Radius of the target checkpoint
- `dt`: Simulation time step
- `max_steps`: Maximum steps before truncation

## Running the Project

### Training a DDPG Agent

```bash
python agent/DDPG/training_ddpg.py
```

### Training a DQN Agent

```bash
python agent/DQN/training_dqn.py
```

### Testing a Trained Agent

To test the DQN agent:
```bash
python agent/DQN/test_dqn.py
```

### Interactive Play

To manually control the sailboat:
```bash
python environnement/play.py
```

## Project Structure

- `/agent`: Contains the reinforcement learning algorithms
    - `/DQN`: Deep Q-Network implementation
    - `/DDPG`: Deep Deterministic Policy Gradient implementation
- `/environnement`: Contains the sailboat simulation environment
- `/models`: Saved trained models
- `/logs`: Training logs

## Implementation Details

### DQN Agent

The DQN agent discretizes the action space into 9 possible actions (combinations of sail and rudder settings). It uses:
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration strategy

### DDPG Agent

The DDPG agent uses a continuous action space with:
- Actor-critic architecture
- Soft target updates
- Exploration noise

### Visualization

The environment includes a rendering system that visualizes:
- The sailboat position and orientation
- Sail and rudder positions
- Wind direction
- Target checkpoint

## Results

Both algorithms successfully learn to navigate the sailboat to the checkpoint. The DDPG agent tends to learn more efficient sailing strategies due to its continuous action space.

## License

This project is provided for educational purposes.