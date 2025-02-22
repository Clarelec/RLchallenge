import torch
import matplotlib.pyplot as plt

from env import Env

env = Env(1)

def test_env():
    action = torch.tensor([[0,0]])
    state = env.reset()
    X = []
    Y = []
    for i in range(5):
        X.append(state[0,0].item())
        Y.append(state[0,1].item())
        if i >0 :
            action = torch.tensor([[0.5,0.5]])
        new_state, real_new_state, reward, done = env.step(state,action)
        state = new_state
    plt.plot(X,Y)
    plt.show()
    
test_env()
    
    