# About
Training a pair of competing RL agents to play Tennis using MADDPG

## Tennis Unity Environment

![Tennis](images/tennis)

* Set-up: Two-player game where agents control rackets to bounce ball over a
  net.
* Goal: The agents must bounce ball between one another while not dropping or
  sending ball out of bounds.
* Agents: The environment contains two agent linked to a single Brain named
  TennisBrain. After training you can attach another Brain named MyBrain to one
  of the agent to play against your trained model.
* Agent Reward Function (independent):
  * +0.1 To agent when hitting ball over net.
  * -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
* Brains: One Brain with the following observation/action space.
  * Vector Observation space: 8 variables corresponding to position and velocity
    of ball and racket.
  * Vector Action space: (Continuous) Size of 2, corresponding to movement
    toward net or away from net, and jumping.
  * Visual Observations: None.
* Reset Parameters: One, corresponding to size of ball.
* Benchmark Mean Reward: 2.5
