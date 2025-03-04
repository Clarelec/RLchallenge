import torch
import logging
from environnement.utils import LOG_DIR
import os
import random
LOG_ADRESS = os.path.join(LOG_DIR, os.path.basename(__file__).split('.')[0]+'.log')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename= LOG_ADRESS,
                    filemode='w')

import numpy as np
from environnement.env import Env
from typing import List, Tuple, Callable
import itertools
from tqdm.notebook import tqdm

from collections import namedtuple, deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"device is {device}")

# simple 3 layer MLP network
class QNetwork(torch.nn.Module):
    """
    A Q-Network implemented with PyTorch.
    In the case of this environnement, to transform the output into the action space, we will use the following transformation:
    output = [output//3, output%3] /2
    0 -> [0,0]
    1 -> [0,0.5]
    2 -> [0,1]
    3 -> [0.5,0]
    4 -> [0.5,0.5]
    5 -> [0.5,1]
    6 -> [1,0]
    7 -> [1,0.5]
    8 -> [1,1]
    
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
        Define the forward pass of the QNetwork.
    """

    def __init__(self, n_observations: int =10, n_actions: int= 9, nn_l1: int = 128, nn_l2: int = 128):
        """
        Initialize a new QNetwork instance.

        Args:
            n_observations (int, optional): Number of observations. Defaults to 10.
            n_actions (int, optional): Number of actions. Defaults to 9.
            nn_l1 (int, optional): size of hidden layer 1. Defaults to 128.
            nn_l2 (int, optional): size of hidden layer 2. Defaults to 128.
        """
        super(QNetwork, self).__init__()

        self.layer1 = torch.nn.Linear(n_observations, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, n_actions)
        
        logger.info(f"QNetwork initialized with {n_observations} observations and {n_actions} actions")
        
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

        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        output_tensor= self.layer3(x)
        
        
        return output_tensor
    
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
        env: Env,
        q_network: torch.nn.Module,
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
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: torch.Tensor) -> np.int64:
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

        if random.random() < self.epsilon:
            # action = TODO... # Select a random action
            action = torch.Tensor([[random.randint(0, 2), random.randint(0, 2)]])/2
            action.to(device)
            
        else:
            with torch.no_grad():
                
                state_tensor = state.to(device)
               
                q_values = self.q_network(state_tensor)
                
                output = torch.argmax(q_values, dim=1).item()
                # transform the output to the action space
                action = torch.Tensor([[output//3, output%3]])/2
                
        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)




class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_decay: float,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The multiplicative factor of learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : deque
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
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self, *args
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
        self.buffer.append(Transition(*args))
        
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
        batch = Transition(*zip(*random.sample(self.buffer, batch_size)))
        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        # indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # batch = [self.buffer[i] for i in indices]
    
        # states = np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch])
        # actions = np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch])
        # rewards = np.array([x[2] for x in batch])
        # next_states = np.array([x[3].cpu().numpy() if isinstance(x[3], torch.Tensor) else x[3] for x in batch])
        # dones = np.array([x[4].cpu().numpy() if isinstance(x[4], torch.Tensor) else x[4] for x in batch])
    
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



def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, tau:float):
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
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def convert_action(action):
    return round(float(2*(3*action[0] + action[1])))

def train_dqn2_agent(
    env: Env,
    q_network: torch.nn.Module,
    target_q_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epsilon_greedy: EpsilonGreedy,
    device: torch.device,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_episodes: int,
    gamma: float,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    target_q_network_sync_period: int,
) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    target_q_network : torch.nn.Module
        The target Q-network to use for estimating the target Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.
    target_q_network_sync_period : int
        The number of episodes after which the target Q-network should be updated with the weights of the Q-network.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    iteration = 0
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state= env.reset()
        episode_reward = 0.0

        for t in itertools.count():
            # Get action, next_state and reward

            action = epsilon_greedy(state)
            state = state.to(device)
            action = action.to(device)
            
            next_state, real_next_state, reward, truncated, terminated = env.step(state=state, action=action)
           
            done = (terminated | truncated).float().to(device)
            
            converted_action = torch.tensor([convert_action(action.squeeze())], dtype=torch.long, device=device)
            replay_buffer.add(state, converted_action, reward, real_next_state, done)

            episode_reward += float(reward)

            # Update the q_network weights with a batch of experiences from the buffer
            
            if len(replay_buffer) > batch_size:
                batch_states_tensor, batch_actions_tensor, batch_rewards_tensor, batch_next_states_tensor, batch_dones_tensor = replay_buffer.sample(batch_size)

                # Convert to PyTorch tensors
                # batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)

                # batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device).unsqueeze(1)
                # batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                # batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)

                # batch_actions_tensor = torch.tensor([[round(batch_actions[i][0][0]*3*2 + batch_actions[i][0][1]*2)] for i in range(batch_actions.shape[0])], dtype=torch.long, device=device)   
                # print(batch_actions[:10], batch_actions_tensor[:10])
                # Compute the target Q values for the batch
                
                with torch.no_grad():
                       
                    next_state_q_values = target_q_network(batch_next_states_tensor)
                    targets = batch_rewards_tensor + gamma * torch.max(next_state_q_values, dim=1).values * (1 - batch_dones_tensor)

                # Compute Q_value
               
                q_values = q_network(batch_states_tensor.squeeze(1))
                
                current_q_values = torch.gather(q_values, 1, batch_actions_tensor.unsqueeze(1)).squeeze(1)
                # Compute loss
                try:
                    assert current_q_values.shape == targets.shape
                except AssertionError:
                    logger.error(f"Shape mismatch: current_q_values.shape={current_q_values.shape}, targets.shape={targets.shape}")
                    raise
                loss = loss_fn(current_q_values, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()
                logger.info(f"episode {episode_index} step {t} reward {reward} loss {loss.item()} epsilon {epsilon_greedy.epsilon}")
                # Update the target q-network weights

            # Every episodes (e.g., every `target_q_network_sync_period` episodes), the weights of the target network are updated with the weights of the Q-network
            if iteration % target_q_network_sync_period == 0:
                soft_update(local_model=q_network, target_model=target_q_network, tau=5e-3)
                # target_q_network.load_state_dict(q_network.state_dict())
            iteration += 1
            
            # Check if the episode is terminated
            
            if done:
                break

            state = next_state

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()
        print(episode_reward)

    return episode_reward_list




