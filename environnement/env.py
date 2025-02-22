import torch
from math import pi

class Env :
    
    def __init__(self, 
                 batch_size, 
                 max_safran = 20, 
                 max_hull = 20, 
                 checkpoint_radius = 10,
                 mass = 1000,
                 drag = 100,
                 sail = 100,
                 dt = 1,
                 reactivity = 45,
                 max_steps = 200):
        """
        Args:
            batch_size (int) : number of agents
            max_safran (float) : maximum angle change of the safran
            max_hull (float) : maximum angle change of the hull
            checkpoint_radius (float) : radius of the checkpoint
            mass (float) : mass of the boat
            drag (float) : drag coefficient
            sail (float) : sail coefficient (force applied by the sail)
            dt (float) : time step
            reactivity (float) : reactivity of the boat (speed angle change per dt if the safran is orthogonal to the speed)
            max_steps (int) : maximum number of steps before truncation
            
        """
        self.state_dim = 10
        self.action_dim = 2
        self.max_safran = max_safran
        self.max_hull = max_hull
        self.checkpoint_radius = checkpoint_radius
        self.mass = mass
        self.drag = drag
        self.sail = sail
        self.batch_size = batch_size
        self.dt = dt
        self.reactivity = reactivity
        self.max_steps = max_steps
        
        #The checkpoint is placed at the origin
        self.checkpoint = torch.zeros((batch_size, 2))
        
        self.steps = torch.zeros(batch_size)
 
    
    def step(self, state, action):
        """
        Args:
            state (tensor (batch_size,state_dim) ): state of the agent
            action (tensor (batch_size,action_dim) ): action to apply
            
        Outputs:
            (new_state, real_new_state, reward, terminated, truncated)
            
            with 
            
            new_state (tensor (batch_size,state_dim) ): new state of the agent (reseted if needed)
            real_new_state (tensor (batch_size,state_dim) ): real new state (before truncated)
            reward (tensor (batch_size) ) : reward at this step
            dones (tensor (batch_size) ) : 1 if the agent is terminated, 0 otherwise
            
        """
        #We check if state and action have the right shape
        assert action.shape[1] == self.action_dim
        assert state.shape[1] == self.state_dim
        assert state.shape[0] == action.shape[0]
        assert state.shape[0] == self.batch_size
        
        #We extract the different features
        pos = state[:,:2]
        speed = state[:,2:4]
        hull = state[:,4:6]
        safran = state[:,6:8]
        wind = state[:,8:10]
        
        #We update the change in angle of the boat (hull and safran turn with the boat)
        w_boat = self.reactivity * self.sin_angle(safran, speed)
        new_speed = self.rotation(speed, self.dt*w_boat)
        new_safran = self.rotation(safran, self.dt*w_boat)
        new_hull = self.rotation(hull, self.dt*w_boat)
        
        #We compute the new position
        new_pos = pos + new_speed * self.dt
        
        #We extract the actions
        safran_action = action[:,0]
        hull_action = action[:,1]
        
        #We scale the actions
        safran_action = (2*safran_action-1)* self.max_safran
        hull_action = (2*hull_action-1)* self.max_hull
        
        #We update the hull and safran angle
        new_hull = self.rotation(new_hull, hull_action)
        new_safran = self.rotation(new_safran, safran_action)
        
        
        #We compute the force applied by the wind
        v_relat = new_speed - wind
        force = self.sail * torch.sum(new_hull * v_relat, dim=1) * self.rotation(new_hull, torch.ones(self.batch_size)*(-pi/2))
        
        #We project the force on the speed direction
        force = torch.sum(force * new_speed, dim=1) * new_speed / (torch.norm(new_speed, dim=1).unsqueeze(1))**2
        print(force)
        
        #we compute the drag 
        drag = -self.drag * new_speed
        
        #We compute the new speed
        new_speed = new_speed + (force + drag) / self.mass * self.dt
        
        #We concatenate all the features
        real_new_state = torch.cat((new_pos, new_speed, new_hull, new_safran, wind), dim=1)
        
        #We compute the reward
        reward = self.reward(state, action)
        
        #We check if the agent is terminated
        terminated = reward > 0
        truncated = self.steps > self.max_steps
        
        dones = (terminated | truncated).float()
        
        #We update the steps
        self.steps = (self.steps + 1)*(1-dones)
        
        #We reset if needed
        new_state = real_new_state * (1-dones).unsqueeze(1) + self.reset() * dones.unsqueeze(1)
        
        return new_state, real_new_state, reward, dones
        
    
    def reset(self):
        """
        Outputs:
            state (tensor (batch_size,state_dim) ): initial state of the agent
            
        The initial state of the agent is randomly generated in the interval [-100,100] for each dimension
        The three angles are generated in the interval [-pi,pi]
        The initial speed is zero
        
        """
        
        #We generate random initial states
        pos = torch.rand((self.batch_size, 2)) * 200 - 100
        wind = torch.rand((self.batch_size, 1)) * 2 * pi - pi
        speed = torch.rand((self.batch_size, 2)) * 10
        hull = -speed.clone()/torch.norm(speed, dim=1).unsqueeze(1)
        safran = -speed.clone()/torch.norm(speed, dim=1).unsqueeze(1)
        
        #We encode angles with cos and sin to avoid the discontinuity at -pi and pi
        wind = torch.cat((torch.cos(wind), torch.sin(wind)), dim=1)
        
        #We concatenate all the features
        state = torch.cat((pos, speed, hull, safran, wind), dim=1)
        
        return state
        
    
    def reward(self, state, action):
        """
        Args:
            state (tensor (batch_size,state_dim) ): state of the agent
            action (tensor (batch_size,action_dim) ): action to apply
            
        Outputs:
            reward (tensor (batch_size) ) : reward at this step
        """
        #Reward = 100 if the agent is in the checkpoint, -1 otherwise
        dist = self.distance(state[:,:2], state[:,8:10])
        reward = (dist < self.checkpoint_radius)*101 - 1
        return reward
    
    def rotation(self, vector, angle):
        """
        Args:
            angle (tensor (batch_size) ): angle to rotate
            vector (tensor (batch_size,2) ): vector to rotate
            
        Outputs:
            rotated_vector (tensor (batch_size,2) ) : rotated vector
        """
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=1).view(-1, 2, 2)
        rotated_vector = torch.bmm(rotation_matrix, vector.unsqueeze(2)).squeeze(2)
        return rotated_vector
    
    def distance(self,pos1, pos2):
        """
        Args:
            pos1 (tensor (batch_size,2) ): position 1
            pos2 (tensor (batch_size,2) ): position 2
            
        Outputs:
            distance (tensor (batch_size) ) : distance between pos1 and pos2
        """
        return torch.norm(pos1 - pos2, dim=1)
        
    def sin_angle(self,A, B):
        """
        Args:
            A (tensor (batch_size,2) ): vector A
            B (tensor (batch_size,2) ): vector B
            
        Outputs:
            sin_angle (tensor (batch_size) ) : sin of the angle between A and B
        """
        cross_prod = torch.abs(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])
        norm_A = torch.norm(A, dim=1)
        norm_B = torch.norm(B, dim=1)
        return cross_prod / (norm_A * norm_B)
        
        
        
        
        