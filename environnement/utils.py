def state_to_dict(state):
    pos = state[:,:2]
    speed = state[:,2:4]
    hull = state[:,4:6]
    safran = state[:,6:8]
    wind = state[:,8:10]
    
    return {pos : pos,
            speed : speed,
            hull : hull,
            safran : safran,
            wind : wind 
            }


    