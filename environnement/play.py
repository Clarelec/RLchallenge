from env import  Env
import torch
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(batch_size=1, dt=0.1, max_steps=100, device=device)

done = False
state = env.reset()
env.render(state)

while not done :
    
    sail = float(input("Enter the sail action: "))
    safran= float(input("Enter the safran action: "))
    action = torch.Tensor([[sail, safran]]).to(device)
    state, _,reward, truncated, terminated = env.step(state,action)
    done = terminated or truncated
    time.sleep(0.1)
    env.render(state)