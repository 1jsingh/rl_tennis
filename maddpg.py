import numpy as np
import random
import copy
from collections import namedtuple, deque

from model4 import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4       # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-1        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents=2,random_seed=0):
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
            self.actor_optimizer.append(optim.Adam(self.actor_local[i].parameters(), lr=LR_ACTOR))

            # Critic Network (w/ Target Network)
            self.critic_local.append(Critic(state_size, action_size, random_seed).to(device))
            self.critic_target.append(Critic(state_size, action_size, random_seed).to(device))
            self.critic_optimizer.append(optim.Adam(self.critic_local[i].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY))

            
        # Noise process
        self.noise = OUNoise((self.num_agents,action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        #self.soft_update(self.critic_local[0], self.critic_target[0], 1.0)
        #self.soft_update(self.actor_local[0], self.actor_target[0], 1.0)

        #self.soft_update(self.critic_local[1], self.critic_target[1], 1.0)
        #self.soft_update(self.actor_local[1], self.actor_target[1], 1.0)
    
    def step(self,state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state[0], action[0], reward[0], next_state[0], done[0],state[1], action[1], reward[1], next_state[1], done[1])
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        actions = np.zeros((self.num_agents,self.action_size))
        for i in range(self.num_agents):
            self.actor_local[i].eval()
            with torch.no_grad():
                action = self.actor_local[i](state[i]).cpu().data.numpy()
                actions[i,:] = action
            self.actor_local[i].train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

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
        states1, actions1,rewards1, next_states1, dones1,states2, actions2,rewards2, next_states2, dones2  = experiences

        # ---------------------------- get actions_next -------------------------- #

        actions_next1 = self.actor_target[0](next_states1)
        actions_next2 = self.actor_target[1](next_states2)

        # ---------------------------- update critic1 ---------------------------- #
        # Get predicted Q values from target models
        Q_targets_next1 = self.critic_target[0](next_states1, actions_next1, next_states2, actions_next2)
        # Compute Q targets for current states (y_i)
        Q_targets1 = rewards1 + (gamma * Q_targets_next1 * (1 - dones1))
        # Compute critic loss
        Q_expected1 = self.critic_local[0](states1, actions1, states2, actions2)
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets1)
        # Minimize the loss
        self.critic_optimizer[0].zero_grad()
        critic_loss1.backward(retain_graph=True)
        self.critic_optimizer[0].step()

        # ---------------------------- update critic2 ---------------------------- #
        # Get predicted Q values from target models
        Q_targets_next2 = self.critic_target[1](next_states2, actions_next2, next_states1, actions_next1)
        # Compute Q targets for current states (y_i)
        Q_targets2 = rewards2 + (gamma * Q_targets_next2 * (1 - dones2))
        # Compute critic loss
        Q_expected2 = self.critic_local[1](states2, actions2, states1, actions1)
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets2)
        # Minimize the loss
        self.critic_optimizer[1].zero_grad()
        critic_loss2.backward()
        self.critic_optimizer[1].step()


        # ---------------------------- get actions_pred ---------------------------- #
        actions_pred1 = self.actor_local[0](states1)
        actions_pred2 = self.actor_local[1](states2)

        # ---------------------------- update actor1 ---------------------------- #
        # Compute actor loss
        actor_loss1 = -self.critic_local[0](states1, actions_pred1, states2, actions_pred2).mean()
        # Minimize the loss
        self.actor_optimizer[0].zero_grad()
        actor_loss1.backward(retain_graph=True)
        self.actor_optimizer[0].step()

        # ---------------------------- update actor2---------------------------- #
        # Compute actor loss
        actor_loss2 = -self.critic_local[0](states1, actions_pred1, states2, actions_pred2).mean()
        # Minimize the loss
        self.actor_optimizer[1].zero_grad()
        actor_loss2.backward()
        self.actor_optimizer[1].step()

        # ----------------------- update target1 networks ----------------------- #
        self.soft_update(self.critic_local[0], self.critic_target[0], TAU)
        self.soft_update(self.actor_local[0], self.actor_target[0], TAU)

        # ----------------------- update target2 networks ----------------------- #
        self.soft_update(self.critic_local[1], self.critic_target[1], TAU)
        self.soft_update(self.actor_local[1], self.actor_target[1], TAU)


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

    def __init__(self, size, seed, mu=0., theta=0.15,sigma_start=1.0,sigma_decay= .9999,sigma_end=.1):
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
        self.experience = namedtuple("Experience", field_names=["state1", "action1", "reward1", "next_state1", "done1", "state2", "action2", "reward2", "next_state2", "done2"])
        self.seed = random.seed(seed)
    
    def add(self, states1, actions1,rewards1, next_states1, dones1,states2, actions2,rewards2, next_states2, dones2):
        """Add a new experience to memory."""
        e = self.experience(states1, actions1,rewards1, next_states1, dones1,states2, actions2,rewards2, next_states2, dones2)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states1 = torch.from_numpy(np.vstack([e.state1 for e in experiences if e is not None])).float().to(device)
        actions1 = torch.from_numpy(np.vstack([e.action1 for e in experiences if e is not None])).float().to(device)
        rewards1 = torch.from_numpy(np.vstack([e.reward1 for e in experiences if e is not None])).float().to(device)
        next_states1 = torch.from_numpy(np.vstack([e.next_state1 for e in experiences if e is not None])).float().to(device)
        dones1 = torch.from_numpy(np.vstack([e.done1 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        states2 = torch.from_numpy(np.vstack([e.state2 for e in experiences if e is not None])).float().to(device)
        actions2 = torch.from_numpy(np.vstack([e.action2 for e in experiences if e is not None])).float().to(device)
        rewards2 = torch.from_numpy(np.vstack([e.reward2 for e in experiences if e is not None])).float().to(device)
        next_states2 = torch.from_numpy(np.vstack([e.next_state2 for e in experiences if e is not None])).float().to(device)
        dones2 = torch.from_numpy(np.vstack([e.done2 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states1, actions1,rewards1, next_states1, dones1,states2, actions2,rewards2, next_states2, dones2)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)