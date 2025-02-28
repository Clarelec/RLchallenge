import torch
import sys
import os
import logging

LOG_DIR = r".\logs"
MODEL_DIR = r".\models"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

from environnement import Env
from DDPG import DDPGAgent, train_ddpg



env = Env()
agent = DDPGAgent(env.state_dim, env.action_dim)
train_ddpg(agent, env)


