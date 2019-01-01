## Learning Algorithm

3 separate learning algorithms were implemented for solving this environment.
* `ddpg`: single ddpg agent with shared actor and critic models
* `ddpg_multi`: multiple DDPG agents with separate actor and critic models
* `maddpg`: implementation of the MADDPG algorithm with separate DDPG agents, where the centralized critic has access to combined action and observation states of all agents

### Model Architectures for multiple DDPG algorithm

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
`BATCH_SIZE = 512        # minibatch size`    
`GAMMA = 0.99            # discount factor`   
`TAU = 2e-1              # for soft update of target parameters`   
`LR_ACTOR = 1e-4         # learning rate of the actor`     
`LR_CRITIC = 1e-3        # learning rate of the critic`   
`WEIGHT_DECAY = 0        # L2 weight decay`   


### Learning curve
The environment was solved in 2863 episodes!       
The Avg reward for last 100 episodes = 0.9036

![reward_plot](images/reward_plot_ddpg_multi)

### Future directions for improvement
* ~~Use a prioritised experience replay~~
* Add noise to Actor, Critic hyperparameters
* Use [Curiosity driven exploration](https://pathak22.github.io/large-scale-curiosity/) strategies
