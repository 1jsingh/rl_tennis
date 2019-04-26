import numpy as np
import random
import copy
from collections import namedtuple, deque

from agents.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


WEIGHT_DECAY =  0        # L2 weight decay
UPDATE_EVERY = 1         # step size for agent update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,num_agents, num_parallel_env,agent_indices, buffer_size=int(1e6), 
                            batch_size=128, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, random_seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.num_parallel_env = num_parallel_env
        self.agent_indices = agent_indices


        # define agent hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.actor_local = []
        self.actor_target = []
        self.actor_optimizer = []

        self.critic_local = []
        self.critic_target = []
        self.critic_optimizer = []

        for i in range(self.num_agents):
            # Actor Network (w/ Target Network)
            self.actor_local.append(Actor(state_size, action_size, random_seed).to(device))
            self.actor_target.append(Actor(state_size, action_size, random_seed).to(device))
            self.actor_optimizer.append(optim.Adam(self.actor_local[i].parameters(), lr=lr_actor))

            # Critic Network (w/ Target Network)
            self.critic_local.append(Critic(state_size, action_size, random_seed).to(device))
            self.critic_target.append(Critic(state_size, action_size, random_seed).to(device))
            self.critic_optimizer.append(optim.Adam(self.critic_local[i].parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY))

            #self.soft_update(self.critic_local[i], self.critic_target[i], 1.0)
            #self.soft_update(self.actor_local[i], self.actor_target[i], 1.0)

            
        # Noise process
        self.noise = OUNoise((self.num_agents*self.num_parallel_env,action_size),random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)

        self.t = 0
    
    def step(self,state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        for i in range(self.num_parallel_env):
            same_env_agent_indices = list(self.agent_indices[i])
            states = state[same_env_agent_indices]
            actions = action[same_env_agent_indices]
            rewards = reward[same_env_agent_indices]
            next_states = next_state[same_env_agent_indices]
            dones = done[same_env_agent_indices]
            self.memory.add(states, actions,rewards, next_states, dones)

        self.t+=1    
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and self.t%UPDATE_EVERY == 0:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        actions = np.zeros((states.shape[0],self.action_size))
        for i in range(self.num_agents):
            self.actor_local[i].eval()
            with torch.no_grad():
                action = self.actor_local[i](states[i*self.num_parallel_env: (i+1)*self.num_parallel_env]).cpu().data.numpy()
                actions[i*self.num_parallel_env: (i+1)*self.num_parallel_env] = action
            self.actor_local[i].train()
        
        if add_noise:
           actions += self.noise.sample()
        
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()
        self.t = 0

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions,rewards, next_states, dones  = experiences

        # ---------------------------- get actions_next and actions_pred -------------------------- #
        actions_pred_next = []
        #actions_pred_current = []
        #log_probs=[]
        
        with torch.no_grad():
            for i in range(self.num_agents):
                actions_pred_next.append(self.actor_target[i](next_states[i]))
                #actions_pred = self.actor_local[i](states[i])
                #actions_pred_current.append(actions_pred)
                #log_probs.append(log_prob)

            actions_pred_next = torch.cat(tuple(actions_pred_next),dim=-1)
            #actions_pred_current = torch.cat(tuple(actions_pred_current),dim=-1)

        for i in range(self.num_agents):
            # ---------------------------- update critic ---------------------------- #
            # Get predicted Q values from target models
            Q_targets_next = self.critic_target[i](torch.cat(tuple(next_states),dim=-1), actions_pred_next).squeeze()
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[i] + (gamma * Q_targets_next * (1 - dones[i]))
            # Compute critic loss
            Q_expected = self.critic_local[i](torch.cat(tuple(states),dim=-1),torch.cat(tuple(actions),dim=-1)).squeeze()
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            
            # Minimize the loss
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            #nn.utils.clip_grad_norm_(self.critic_local[i].parameters(), 0.5)
            self.critic_optimizer[i].step()

            # ---------------------------- update actor ---------------------------- #
            actions_ = actions.clone()
            actions_[i] = self.actor_local[i](states[i])
            # Compute actor loss
            actor_loss = -self.critic_local[i](torch.cat(tuple(states),dim=-1),torch.cat(tuple(actions_),dim=-1)).squeeze().mean()
            
            #q_values = self.critic_local[i](torch.cat(tuple(states),dim=-1),actions_pred_current)
            #actor_loss = -(log_probs[i]*q_values).mean()
            
            # Minimize the loss
            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            #nn.utils.clip_grad_norm_(self.actor_local[i].parameters(), 0.5)
            self.actor_optimizer[i].step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local[i], self.critic_target[i], self.tau)
            self.soft_update(self.actor_local[i], self.actor_target[i], self.tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15,sigma_start=1.0,sigma_decay=.9999,sigma_end=.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma_start
        self.sigma_decay = sigma_decay
        self.sigma_end = sigma_end
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)   #np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        # decay sigma
        self.sigma = max(self.sigma*self.sigma_decay,self.sigma_end)
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions,rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions,rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None],axis=1)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None],axis=1)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None],axis=1)).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None],axis=1)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None],axis=1).astype(np.uint8)).float().to(device)

        return (states, actions,rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)