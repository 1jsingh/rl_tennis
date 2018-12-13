## Learning Algorithm

The learning algorithm is a Distributed Training DDPG algorithm which uses multiple(20) parallel, non interacting copies of agents
to distribute the task of gathering experience.

### Model Architectures

#### Actor

`fc1 = [Fully connected, dim = 400, RELU](states)`       
`fc2 = [Fully connected, dim = 300, RELU](fc1)`        
`actions = [Fully connected, dim = action_size, tanh](fc2)`      

#### Critic

`fc1 = [Fully connected, dim = 400, RELU](states)`    
`fc_comb = CONCAT(fc1,actions)`   
`fc2 = [Fully connected, dim = 300, RELU](fc_comb)`    
`Q_value = [Fully connected, dim = 1, activation = None](fc2)`    

#### DDPG hyperparameters

`BUFFER_SIZE = int(1e5)  # replay buffer size`    
`BATCH_SIZE = 128        # minibatch size`    
`GAMMA = 0.99            # discount factor`   
`TAU = 1e-3              # for soft update of target parameters`   
`LR_ACTOR = 1e-4         # learning rate of the actor`     
`LR_CRITIC = 1e-3        # learning rate of the critic`   
`WEIGHT_DECAY = 0        # L2 weight decay`   


### Learning curve
The environment was solved in just 87 episodes!       
The Avg reward for last 100 episodes = 34.27 

![reward_plot](images/reward_plot)

### Future directions for improvement
* Use a prioritised experience replay
* Add noise to Actor, Critic hyperparameters
* Use separate ddpg agents with a centralized critic
