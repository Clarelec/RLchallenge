import torch
import os
import sys


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.DDPG.DDPG import DDPGAgent, train_ddpg
from environnement.env import Env
import torch
import time


env = Env(64, dt = 0.1)
agent = DDPGAgent(env.state_dim,env.action_dim)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent.actor.load_state_dict(torch.load('models/ddpg_actor_network.pth', map_location=device))
agent.actor.to(device)

def test_env():
    state = env.reset()

    for i in range(200):
        action = agent.act(state)
        new_state, real_new_state, reward, terminated, truncated = env.step(state,action)
        state = new_state
        env.render(state)
        time.sleep(0.1)
    
    
test_env()
    
    