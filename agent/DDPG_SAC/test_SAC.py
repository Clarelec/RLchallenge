import torch
import os
import sys


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.DDPG_SAC.DDPG_SAC import DDPGAgent, train_ddpg
from environnement.env import Env
import torch
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(64, dt = 0.1)
agent = DDPGAgent(env.state_dim,env.action_dim)



network =torch.load('models/SAC_actor_network.pth', map_location=device).to(device)
network.eval()

agent = DDPGAgent(env.state_dim,env.action_dim)
agent.actor = network
agent.is_training = False
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
    
    