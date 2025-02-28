import torch
import time
from env import Env
env = Env(64, dt = 0.1)

def test_env():
    action = torch.tensor([[1,1]])*torch.ones((64,2))
    state = env.reset()

    for i in range(50):
        print(i)
        if i >5 :
            action = torch.tensor([[0.5,0.5]])*torch.ones((64,2))
        new_state, real_new_state, reward, terminated, truncated = env.step(state,action)
        state = new_state
        env.render(state)
        time.sleep(0.1)
    
    
test_env()
    
    