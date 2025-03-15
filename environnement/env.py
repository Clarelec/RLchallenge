import torch
from math import pi
from render import RENDER_ENV

class Env :
    
    def __init__(self, 
                 batch_size = 64, 
                 max_safran = pi/16, 
                 max_sail = pi/16, 
                 checkpoint_radius = 50,
                 mass = 1000,
                 drag = 100,
                 sail = 500,
                 wind = 100,
                 dt = 0.1,
                 reactivity = pi/8,
                 max_steps = 200,
                 render_width = 800,
                 render_height= 400,
                 device = 'cpu',
                 incentive = True,
                 incentive_coeff = 0.01,
                 render_needed = True,
                 spaun_size = 1000
                 ):
        """
        Args:
            batch_size (int) : number of envs to run in parallel
            max_safran (float) : maximum angle change of the safran
            max_sail (float) : maximum angle change of the sail
            checkpoint_radius (float) : radius of the checkpoint
            mass (float) : mass of the boat
            drag (float) : drag coefficient
            sail (float) : sail coefficient (force applied by the sail)
            wind (float) : wind speed
            dt (float) : time step
            reactivity (float) : reactivity of the boat (speed angle change per dt if the safran is orthogonal to the speed)
            max_steps (int) : maximum number of steps before truncation
            render_width (int) : width of the canvas for the visualization
            render_height (int) : height of the canvas for the visualization
        """
        self.state_dim = 10
        self.action_dim = 2
        self.max_safran = max_safran
        self.max_sail = max_sail
        self.checkpoint_radius = checkpoint_radius
        self.mass = mass
        self.drag = drag
        self.sail = sail
        self.wind = wind
        self.batch_size = batch_size
        self.dt = dt
        self.reactivity = reactivity
        self.max_steps = max_steps
        self.device = device
        self.incentive = incentive
        self.incentive_coeff = incentive_coeff
        #The checkpoint is placed at the origin
        self.checkpoint = torch.zeros((batch_size, 2)).to(self.device)
        self.spaun_size = spaun_size
        self.steps = torch.zeros(batch_size).to(self.device)     
        if render_needed:
            self._renderer = RENDER_ENV(
                width=render_width, 
                height=render_height,
                checkpoint_radius=checkpoint_radius
            )
            
        self.reset() # pour initialiser d'autres attributs
    
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
        assert state.ndim == 2
        assert action.ndim == 2
        
        state = state.to(self.device)
        action = action.to(self.device)
        
        previous_distance = torch.sqrt(torch.sum(state[:, 0:2]**2, dim=1))
        
        
        
        #We extract the different features
        pos = state[:,:2]
        speed = state[:,2:4]
        sail = state[:,4:6]
        safran = state[:,6:8]
        wind = state[:,8:10]
        
        #We update the change in angle of the boat (sail and safran turn with the boat)
        w_boat = self.reactivity * self.sin_angle(safran, speed)
        new_speed = self.rotation(speed, self.dt*w_boat)
        new_safran = self.rotation(safran, self.dt*w_boat)
        new_sail = self.rotation(sail, self.dt*w_boat)
        
        #We compute the new position
        new_pos = pos + new_speed * self.dt
        
        #We extract the actions
        safran_action = action[:,0]
        sail_action = action[:,1]
        
        #We scale the actions
        safran_action = (2*safran_action-1)* self.max_safran
        sail_action = (2*sail_action-1)* self.max_sail
        
        #We update the sail and safran angle
        new_sail2 = self.rotation(new_sail, sail_action)
        new_safran2 = self.rotation(new_safran, safran_action)
        
        # We only allow actions that keep both safran and sail in the opposite direction of the boat
        mask = (new_sail2 * new_speed).sum(dim=1) < 0
        new_sail[mask] = new_sail2[mask]
        
        mask = (new_safran2 * new_speed).sum(dim=1) < 0
        new_safran[mask] = new_safran2[mask]
        
        #We compute the force applied by the wind
        v_relat = self.wind * wind - new_speed
        new_speed_normal = self.rotation(new_speed, torch.ones(self.batch_size, device= self.device)*(pi/2))
        
        # The following lines change the sail direction if the wind is in the opposite direction
        # Result of the sail change is stored in the variable reflected_sail
        speed_norm = torch.nn.functional.normalize(new_speed, dim=1)
        projection = (new_sail * speed_norm).sum(dim=1, keepdim=True) * speed_norm
        reflected_sail = 2 * projection - new_sail
        
        # We check if a sail change is needed
        dot_product = (v_relat * new_speed_normal).sum(dim=1, keepdim=True)
        wind_orientation = dot_product * new_speed_normal
        dot_product2 = (wind_orientation * new_sail).sum(dim=1)
        
        # We apply the sail change if needed
        mask = dot_product2 < 0
        new_sail[mask] = reflected_sail[mask]
        
        angles = -pi/2 * (dot_product > 0) + pi/2 * (dot_product <= 0)
        new_sail_normal = self.rotation(new_sail, angles)
        
        # We compute the force applied by the sail
        force = self.sail * torch.max(torch.zeros(self.batch_size,1, device=self.device),torch.sum(new_sail_normal * v_relat, dim=1, keepdim=True)) * new_sail_normal
        
        #We project the force on the speed direction
        force = torch.sum(force * new_speed, dim=1, keepdim=True) * new_speed /(1e-3+(torch.norm(new_speed, dim=1).unsqueeze(1))**2)

        #we compute the drag 
        drag = -self.drag * new_speed
        
        #We compute the new speed
        new_speed = new_speed + (force + drag) / self.mass * self.dt
        
        #We concatenate all the features
        real_new_state = torch.cat((new_pos, new_speed, new_sail, new_safran, wind), dim=1).to(self.device)
        
        
        #We check if the agent is terminated
        terminated = torch.sqrt(torch.sum(new_pos**2, dim=1))<self.checkpoint_radius
        truncated = self.steps > self.max_steps
                
        dones = (terminated | truncated).float().to(self.device)
        
        #We update the steps
        self.steps = (self.steps + 1)*(1-dones)
        
        #We reset if needed
        dones=dones.unsqueeze(1)
        new_state = (1-dones)*real_new_state +dones* self.reset()
        new_state = new_state.to(self.device)
        
        current_distance = torch.sqrt(torch.sum(real_new_state[:, 0:2]**2, dim=1))


        #We compute the reward
        reward = self.reward(previous_distance, current_distance)
        
        return new_state, real_new_state, reward, terminated, truncated
        
    
    def reset(self):
        """
        Outputs:
            state (tensor (batch_size,state_dim) ): initial state of the agent
            
        The initial state of the agent is randomly generated in the interval [-100,100] for each dimension
        The three angles are generated in the interval [-pi,pi]
        The initial speed is zero
        
        """
        
        #We generate random initial states
        verif = True
        while  verif:
            pos = torch.rand((self.batch_size, 2)) * self.spaun_size - self.spaun_size/2
            verif = False
            if torch.norm(pos, dim=1).min() < self.checkpoint_radius :
                verif = True 
        wind = torch.rand((self.batch_size, 1)) * 2 * pi - pi
        speed = torch.rand((self.batch_size, 2)) * 10
        sail = -speed.clone()/torch.norm(speed, dim=1).unsqueeze(1)
        safran = -speed.clone()/torch.norm(speed, dim=1).unsqueeze(1)
        
        #We encode angles with cos and sin to avoid the discontinuity at -pi and pi
        wind = torch.cat((torch.cos(wind), torch.sin(wind)), dim=1)
        
        #We concatenate all the features
        state = torch.cat((pos, speed, sail, safran, wind), dim=1).to(self.device)
                
        return state
        
    
    def reward(self, previous_distance, current_distance):
        """
        Args:
            state (tensor (batch_size,state_dim) ): state of the agent
            action (tensor (batch_size,action_dim) ): action to apply
            
        Outputs:
            reward (tensor (batch_size) ) : reward at this step
        """
        #Reward = 100 if the agent is in the checkpoint, -1 otherwise
        # dist = self.distance(state[:,:2], state[:,8:10])
        # reward = (dist < self.checkpoint_radius)*101 - 1
        # reward = reward.to(self.device)
        # if self.incentive:
        #     reward = reward - self.incentive_coeff * torch.norm(state[:,:2], dim=1)

        return (previous_distance-current_distance)/100
    
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
        rotation_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=1).view(-1, 2, 2).to(self.device)
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
        return torch.norm(pos1 - pos2, dim=1).to(self.device)
        
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
        res =  cross_prod / (norm_A * norm_B)
        res= res.to(self.device)
        return res
        
        
        
    def render(self,state: torch.Tensor, agent_index : int = 0):
        """
        Args:
            state (tensor (batch_size,state_dim) ): state of the agent
            agent_index (int) : if batch_size>0, number of the agent to render
        Outputs:
            None
            
        The function displays the state of the agent
        """
        self._renderer.afficher_etat(state, self.checkpoint)
            
        