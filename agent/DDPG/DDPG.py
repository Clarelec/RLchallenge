import torch
import logging
from environnement.utils import LOG_DIR
import os
import numpy as np
from typing import Tuple
import collections
from tqdm import tqdm
import random
import mlflow

LOG_ADRESS = os.path.join(LOG_DIR, os.path.basename(__file__).split('.')[0]+'.log')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename= LOG_ADRESS,
                    filemode='w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"device is {device}")

class Actor(torch.nn.Module):
    """
    An Actor implemented with PyTorch.
    
    Attributes
    ----------
    layer1 : torch.nn.Linear
        First fully connected layer.
    layer2 : torch.nn.Linear
        Second fully connected layer.
    layer3 : torch.nn.Linear
        Third fully connected layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Define the forward pass of the Actor.
    """

    def __init__(self, dim_observations: int =10, dim_actions: int= 2, nn_l1: int = 256, nn_l2: int = 256):
        """
        Initialize a new Actor instance.

        Args:
            n_observations (int, optional): Number of observations. Defaults to 10.
            n_actions (int, optional): Number of actions. Defaults to 9.
            nn_l1 (int, optional): size of hidden layer 1. Defaults to 128.
            nn_l2 (int, optional): size of hidden layer 2. Defaults to 128.
        """
        super(Actor, self).__init__()

        self.layer1 = torch.nn.Linear(dim_observations**2, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, dim_actions)
        
        logger.info(f"Actor initialized with observation of dim {dim_observations} and action of dim {dim_actions}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the QNetwork.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (state).

        Returns
        -------
        torch.Tensor
            The output tensor (Q-values).
        """
        x = x.unsqueeze(2)
        x = torch.bmm(x, x.transpose(1,2))

        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        action = torch.nn.functional.sigmoid(self.layer3(x))
        return action
    
class Critic(torch.nn.Module):
    """
    An Critic implemented with PyTorch.
    
    Attributes
    ----------
    layer1 : torch.nn.Linear
        First fully connected layer.
    layer2 : torch.nn.Linear
        Second fully connected layer.
    layer3 : torch.nn.Linear
        Third fully connected layer.

    Methods
    -------
    forward(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor
        Define the forward pass of the Actor.
    """

    def __init__(self, dim_observations: int =10, dim_actions: int= 2, nn_l1: int = 256, nn_l2: int = 256):
        """
        Initialize a new Actor instance.

        Args:
            n_observations (int, optional): Number of observations. Defaults to 10.
            n_actions (int, optional): Number of actions. Defaults to 9.
            nn_l1 (int, optional): size of hidden layer 1. Defaults to 128.
            nn_l2 (int, optional): size of hidden layer 2. Defaults to 128.
        """
        super(Critic, self).__init__()

        self.layer1 = torch.nn.Linear(dim_observations+dim_actions, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, 1)
        
        logger.info(f"Critic initialized with observation of dim {dim_observations} and action of dim {dim_actions}")
        
    def forward(self, state: torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the QNetwork.

        Parameters
        ----------
        state : torch.Tensor
        action : torch.Tensor

        Returns
        -------
        torch.Tensor
            The output tensor (Q-value).
        """

        x = torch.cat([state, action], dim=1)
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        value= self.layer3(x)
        
        
        return value

class DDPGAgent:
    """
    A DDPG Agent.
    
    Attributes
    ----------
    actor : Actor
        The actor network.
    critic : Critic
        The critic network.
    actor_target : Actor
        The target actor network.
    critic_target : Critic
        The target critic network.
    actor_optimizer : torch.optim.Adam
        The optimizer for the actor network.
    critic_optimizer : torch.optim.Adam
        The optimizer for the critic network.
    gamma : float
        The discount factor.
    tau : float
        The target network update rate.
    batch_size : int
        The batch size.
    replay_buffer : ReplayBuffer
        The replay buffer.
    device : torch.device
        The device to use for PyTorch computations.
    is_training : bool
        Whether the agent is training or not.
    """
    
    def __init__(self, state_dim : int,
                 action_dim : int,
                 replay_buffer_size : int = 100000,
                 gamma = 0.99,
                 tau = 0.005,
                 batch_size = 256,
                 device = device,
                 action_noise = 0.1,):
        """
        Initialize a new DDPG Agent.
        """
        # Initialize the actor and critic networks
        self.actor = Actor(dim_observations=state_dim, dim_actions=action_dim).to(device)
        self.actor_target = Actor(dim_observations=state_dim, dim_actions=action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        
        self.critic = Critic(dim_observations=state_dim, dim_actions=action_dim).to(device)
        self.critic_target = Critic(dim_observations=state_dim, dim_actions=action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Initialize the hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = TorchReplayBuffer(replay_buffer_size)
        self.device = device
        self.is_training = True
        self.action_noise = action_noise
    
    def act(self, state : torch.Tensor) -> torch.Tensor:
        """
        Select an action for the given state.

        Parameters
        ----------
        state : torch.Tensor
            The current state.
        Returns
        -------
        torch.Tensor
            The selected action.
        """
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        if self.is_training:
            action += np.random.normal(0, self.action_noise, size=action.shape)
            action = np.clip(action, 0, 1)
        return torch.Tensor(action)
    
    def update_agent(self):
        """
        Update the agent using a batch of experiences from the replay buffer.

        Parameters
        ----------
        buffer_sample : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A batch of experiences from the replay buffer.
        """
        states, actions, rewards, next_states, terminated = self.replay_buffer.sample(self.batch_size)
        
        # Update the critic
        next_actions = self.actor_target(next_states)
        target_q_values = self.critic_target(next_states, next_actions).detach().squeeze()
        target_values = rewards + (1 - terminated.float()) * self.gamma * target_q_values
        predicted_values = self.critic(states, actions).squeeze()
        critic_loss = torch.nn.functional.mse_loss(predicted_values, target_values)

        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update the actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        mlflow.log_metric("actor_loss", actor_loss)
        mlflow.log_metric("critic_loss", critic_loss)
        
    
    def update_target_networks(self):
        """
        Update the target networks.
        """
        # Update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module):
        """
        Soft-update: param_target = tau*param_local + (1 - tau)*param_target
        
        Parameters
        ----------
        local_model : torch.nn.Module
            The local model.
        target_model : torch.nn.Module
            The target model.
            
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """
        Add a new transition to the replay buffer.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action : np.ndarray
            The action taken.
        reward : np.ndarray
            The reward received.
        next_state : np.ndarray
            The next state.
        done : np.ndarray
            Whether the episode is done or not.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)


class TorchReplayBuffer:
    """
    A Replay Buffer that stores torch tensors
    
    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.
    add(state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Sample a batch of transitions from the buffer.
    """
    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer: collections.deque = collections.deque(maxlen=capacity)
    
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor):
        """
        Add a new transition to the buffer.

        Args:
            state (torch.Tensor (batch_size,state_dim)): The state vector of the added transition.
            action (torch.Tensor (batch_size,action_dim)): The action of the added transition.
            reward (torch.Tensor (batch_size,1)): The reward of the added transition.
            next_state (torch.Tensor (batch_size,state_dim)): The next state vector of the added transition.
            done (torch.Tensor (batch_size,1)): dones

        Returns:
            None
        """
        for i in range(state.shape[0]):
            self.buffer.append((state[i], action[i], reward[i], next_state[i], done[i]))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A batch of `batch_size` transitions.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.stack([x[0] for x in batch])
        actions = torch.stack([x[1] for x in batch])
        rewards = torch.stack([x[2] for x in batch])
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.stack([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    
    
    
class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.int64,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, Tuple[int], Tuple[float], np.ndarray, Tuple[bool]]:
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        # Here, `random.sample(self.buffer, batch_size)`
        # returns a list of tuples `(state, action, reward, next_state, done)`
        # where:
        # - `state`  and `next_state` are numpy arrays
        # - `action` and `reward` are floats
        # - `done` is a boolean
        #
        # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
        # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
    
        states = np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch])
        actions = np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3].cpu().numpy() if isinstance(x[3], torch.Tensor) else x[3] for x in batch])
        dones = np.array([x[4].cpu().numpy() if isinstance(x[4], torch.Tensor) else x[4] for x in batch])
    
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return len(self.buffer)

class EpsilonGreedy:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
    ):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def __call__(self, actor_action) -> np.int64:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        np.int64
            The chosen action.
        """
        action = actor_action
        
        if random.random() < self.epsilon:
            action = torch.tensor(np.random.random(action.shape)).float()
            action.to(device)
                
        return action
    
    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def get_base_epsilon_greedy():
    return EpsilonGreedy(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9)

def train_ddpg(ddpgAgent,
               env,
               global_steps = 24_000,
               log_frequency = 800,
               learning_starts = 1000,
               target_update_frequency = 1000,
               epsilon_greedy = get_base_epsilon_greedy()):
    """
    Train a DDPG agent.
    """
    mlflow.start_run()
    state = env.reset()
    total_reward = 0
    total_episodes = 0
    total_terminations = 0
    for step in tqdm(range(global_steps)):
        action = ddpgAgent.act(state)
        action = epsilon_greedy(action)
        state = state.to(device)
        action = action.to(device)
        next_state, real_next_state, reward, terminated, truncated = env.step(state,action)
        ddpgAgent.add_to_buffer(state, action, reward, real_next_state, terminated)
        if step > learning_starts:
            ddpgAgent.update_agent()
            if step % target_update_frequency == 0:
                ddpgAgent.update_target_networks()
        state = next_state
        total_reward += reward.sum()
        total_episodes += terminated.sum() + truncated.sum()
        total_terminations += terminated.sum()
        if step % log_frequency == 0:
            epsilon_greedy.decay_epsilon()
            logger.info(f"Step: {step}, Episodic mean reward : {total_reward/total_episodes}, Epsilon: {epsilon_greedy.epsilon}")
            logger.info(f"Win rate: {total_terminations/total_episodes}")
            print(f"Step: {step}, Episodic mean reward : {total_reward/total_episodes}")
            total_reward = 0
            total_episodes = 0
            total_terminations = 0
            state = env.reset()