import torch
import os
import sys


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))




# Set up logging
import logging
LOG_DIR = r"./logs"
MODEL_DIR = r"./models"

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
from agent.DQN.DQN import QNetwork, ReplayBuffer, MinimumExponentialLR, EpsilonGreedy, train_dqn2_agent, transform_state
from environnement.env import Env
import pandas as pd
import time
from typing import List, Union
import matplotlib.pyplot as plt



logger.info("Libraries imported")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")



def eval_model(env: Env, qnetwork: QNetwork, action_dims: tuple, num_episodes: int = 100) -> List[float]:
    """
    Evaluate a DQN agent on the environment for multiple episodes.
    
    Args:
        env (Env): The environment to evaluate the agent on.
        qnetwork (QNetwork): The Q-Network to use for the agent.
        num_episodes (int): The number of episodes to evaluate the agent.
        
    Returns:
        List[float]: The total rewards for each episode.
    """
    qnetwork.eval()
    rewards = []
    nb_success = 0
    for _ in tqdm(range(num_episodes), desc="Evaluating agent"):
        state = env.reset()
        done = False
        final_reward = 0
        while not done:
            transformed_state = transform_state(state).to(device)
            output = torch.argmax(qnetwork(transformed_state))
            action = torch.Tensor([[output//action_dims[0], output%action_dims[1]]])/2
            action = action.to(device)

            state, _, reward, truncated, terminated = env.step(state, action)
            done = terminated or truncated
            final_reward = reward
        rewards.append(final_reward)
        if terminated:
            nb_success +=1
    qnetwork.train()
    return rewards, nb_success


if __name__ == '__main__':
    reload_q_network=False
    action_dims = 5, 5
    
    
    logger.info("Starting training of DQN2 agent")
    # Initialize environment
    env = Env(batch_size=1, dt=0.1, max_steps=200, device=device, render_needed=False, checkpoint_radius=50)

    logger.info("Environment initialized")

    

    # Initialize Q-Network and target Q-Network
    q_network = QNetwork(n_observations=9, n_actions=action_dims[0]*action_dims[1], nn_l1=128, nn_l2=128  ).to(device)
    if reload_q_network:
        q_network = torch.load(os.path.join(MODEL_DIR, "dqn2_q_network.pth"))
        q_network.to(device)
    
    target_qnetwork = QNetwork(n_observations=9, n_actions=action_dims[0]*action_dims[1], nn_l1=128, nn_l2=128).to(device)
    target_qnetwork.load_state_dict(q_network.state_dict())
    logger.info("Q-Networks initialized and synchronized")

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=1e-3, amsgrad=True)
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.99, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    logger.info("Optimizer, LR scheduler, and loss function initialized")
    
    
    q_network.train()
    target_qnetwork.train()

    # Initialize epsilon-greedy strategy
    epsilon_greedy = EpsilonGreedy(
        epsilon_start=0.9,
        epsilon_min=0.1,
        epsilon_decay=0.99,
        env=env,
        q_network=q_network,
        action_dims=action_dims
    )
    logger.info("Epsilon-greedy strategy initialized")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(20000)
    replay_test_buffer = ReplayBuffer(20000)
    logger.info("Replay buffer initialized")

    # Train the DQN agent
    logger.info("Training DQN2 agent")
    TRAINING_START = time.time()
    episode_reward_list , nb_success= train_dqn2_agent(
            env=env,
            q_network=q_network,
            target_q_network=target_qnetwork,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            replay_buffer=replay_buffer,
            replay_test_buffer=replay_test_buffer,
            epsilon_greedy=epsilon_greedy,
            num_episodes=500,
            gamma=0.95,
            batch_size=2048,
            target_q_network_sync_period=1,
            device=device, 
            action_dims=action_dims
        )
    training_time = time.time() - TRAINING_START
    logger.info(f"Training finished , training time: {training_time:.2f} seconds")
    print(len(np.array(episode_reward_list)[np.array(episode_reward_list)>200]))
    
    print(f" training reussite : {nb_success} / {len(episode_reward_list)}")
    print(f"DQN 2015, final episode reward : {episode_reward_list[-1]}, number of episodes : {len(episode_reward_list)}")
   
    # Evaluate the trained DQN agent
    logger.info("Evaluating trained DQN2 agent")
    EVALUATION_START = time.time()
    evaluation_rewards, nb_success = eval_model(env, q_network, action_dims, num_episodes=100)
    evaluation_time = time.time() - EVALUATION_START
    logger.info(f"Evaluation finished, evaluation time: {evaluation_time:.2f} seconds")
    
    print(nb_success)
    
    # Save the trained Q-Network
    torch.save(q_network, os.path.join(MODEL_DIR, "dqn2_q_network.pth"))
    logger.info("Trained Q-Network saved")



