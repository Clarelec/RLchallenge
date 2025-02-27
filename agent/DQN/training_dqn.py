import torch
import os
import sys


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print(sys.path)



# Set up logging
import logging
LOG_DIR = r".\logs"
MODEL_DIR = r".\models"

LOG_adress = os.path.join(LOG_DIR, os.path.basename(__file__).split('.')[0] + '.log')
print(LOG_adress)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    filename=LOG_adress,
    filemode='w'
)

# Importe libraries
from tqdm.notebook import tqdm
import numpy as np
from agent.DQN.DQN import QNetwork, ReplayBuffer, MinimumExponentialLR, EpsilonGreedy, train_dqn2_agent
from environnement.env import Env
import pandas as pd
import time
from typing import List, Union




logger.info("Libraries imported")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

if __name__ == '__main__':
    
    
    logger.info("Starting training of DQN2 agent")
    # Initialize environment
    env = Env(batch_size=1, dt=0.1, max_steps=300, render_height=200, render_width=400, device=device)
    logger.info("Environment initialized")

    

    # Number of epochs for training
    

    
        

    # Initialize Q-Network and target Q-Network
    q_network = QNetwork(n_observations=env.state_dim, n_actions=9, nn_l1=128, nn_l2=128).to(device)
    target_qnetwork = QNetwork(n_observations=env.state_dim, n_actions=9, nn_l1=128, nn_l2=128).to(device)
    target_qnetwork.load_state_dict(q_network.state_dict())
    logger.info("Q-Networks initialized and synchronized")

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.004, amsgrad=True)
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    logger.info("Optimizer, LR scheduler, and loss function initialized")

    # Initialize epsilon-greedy strategy
    epsilon_greedy = EpsilonGreedy(
        epsilon_start=0.82,
        epsilon_min=0.013,
        epsilon_decay=0.9675,
        env=env,
        q_network=q_network,
    )
    logger.info("Epsilon-greedy strategy initialized")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(2000)
    logger.info("Replay buffer initialized")

    # Train the DQN agent
    logger.info("Training DQN2 agent")
    TRAINING_START = time.time()
    episode_reward_list = train_dqn2_agent(
            env=env,
            q_network=q_network,
            target_q_network=target_qnetwork,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            replay_buffer=replay_buffer,
            epsilon_greedy=epsilon_greedy,
            num_episodes=10,
            gamma=0.9,
            batch_size=128,
            target_q_network_sync_period=30,
            device=device
        )
    training_time = time.time() - TRAINING_START
    logger.info(f"Training finished , training time: {training_time:.2f} seconds")

       

    print(f"DQN 2015, final episode reward : {episode_reward_list[-1]}, number of episodes : {len(episode_reward_list)}")
    logger.info("Training results converted to DataFrame")

    # Save the trained Q-Network
    torch.save(q_network, os.path.join(MODEL_DIR, "dqn2_q_network.pth"))
    logger.info("Trained Q-Network saved")
