import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
LOG_DIR = r".\logs"


LOG_adress = os.path.join(LOG_DIR, os.path.basename(__file__).split('.')[0] + '.log')

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    filename=LOG_adress,
    filemode='w'
)
import time

from environnement.env import Env   
from agent.DQN.DQN import QNetwork
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model_name: str, device: str = 'cpu', **kwargs):
    """
    Test a DQN model on the environment for a single episode.
    Example usage:
    test_model('model.pth', 
    
    env_params={'batch_size': 1, 'dt': 0.1}
    )
    
    Args:
        model_name (str): The name of the model file to load.
            device (str): The device to run the model
        **kwargs: Additional parameters for QNetwork and Env.
    """
    model_path = os.path.join(r".\models", model_name)
    
    
    env_params = kwargs.get('env_params', {})
    
    # Create network and environment
    
    env = Env(**env_params)
    
    # Load model state
    network =torch.load(model_path).to(device)
    network.eval()
    
    state = env.reset()
    
    done = False
    while not done:
        
        output = torch.argmax( network(state), dim=1 ).item()
        action = torch.Tensor([[output//3, output%3]])/2
        
        
        _, state,reward, truncated, terminated = env.step(state,action)
        done = terminated or truncated
        time.sleep(0.1)
        env.render(state)

    print(f"Episode finished, reward: {reward}")
    
    
if __name__ == '__main__':
    test_model('dqn2_q_network.pth', 
               
               env_params={'batch_size': 1, 'dt': 0.1, 'max_steps': 300, 'render_height': 1000, 'render_width': 1000, 'device': device},
                device=device
                )
    