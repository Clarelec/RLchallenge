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
from tqdm import tqdm
import numpy as np
from agent.DDPG.DDPG import DDPGAgent, train_ddpg
from environnement.env import Env
import pandas as pd
import time
from typing import List, Union




logger.info("Libraries imported")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

if __name__ == '__main__':
    
    
    logger.info("Starting training of DDPG agent")
    # Initialize environment
    env = Env(batch_size=256, render_needed=False)
    logger.info("Environment initialized")

    ddpgAgent = DDPGAgent(env.state_dim,env.action_dim)
    logger.info("DDPG agent initialized")
   
    # Train the DQN agent
    logger.info("Training DDPG agent")
    TRAINING_START = time.time()
    train_ddpg(ddpgAgent, env)
    training_time = time.time() - TRAINING_START
    logger.info(f"Training finished , training time: {training_time:.2f} seconds")

    # Save the trained Q-Network
    torch.save(ddpgAgent.actor, os.path.join(MODEL_DIR, "ddpg_actor_network.pth"))
    logger.info("Trained Actor saved")


