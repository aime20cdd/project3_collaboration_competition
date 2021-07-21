# Report

## Learning Algorithm

The chosen algorithm implemented is Multi-Agent Deep Deterministic Policy Gradient (MADDPG). This algorithm extends the DDPG algorithm for use in Multi-Agent problems. Each agent gets it's own Actor/Critic Networks that it utilizes. The key difference in MADDPG, is that the critic is given more information than a typical DDPG agent. In addition to it's own action and observations as inputs, the critic also gets the other agent's actions and observations.

State Space: 24
Action Space: 2

Actor Architecture (States to Actions):
  1. Linear Layer (24 -> 128, ReLU Activation)
  3. Linear Layer (128 -> 128, ReLU Activation)
  4. Linear Layer (128 -> 2, tanh Activation)

Critic Architecture (States & Actions to Reward, includes states and action from both agents):
  1. Linear Layer (48 -> 128, ReLU Activation)
  3. Linear Layer (128 + 4 -> 128, ReLU Activation)
  4. Linear Layer (128 -> 1)

Hyper-Parameters for the Actor-Critic Algorithm are below:
```python
BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

MADDPG(discount_factor=0.95, tau=0.02) # discount_factor = future reward discount factor, tau = soft update of target parameters

noise = 2.0             # Amount of noise to apply
noise_reduction = 0.999 # Amount to reduce the noise by at each episode
```

### MADDPG Explanation

MADDPG is built off of DDPG and uses the same core concepts. See the below section for a description of DDPG. Instead of just having each agent train it's own actor/critic as in DDPG, the observations and actions taken by all agents is included in the critic training for all agents. This helps speed up training since a single agent is going to have to learn how to anticipate what other agent's actions will take to effectively evaluate the value of different actions (this is the Q-table in basic Q-learning and is approximated by the critic network). With the extra information, the critic will be able to get more accurate more quickly. Despite this peak into the global state of the problem, each agent's actor only relies on that agent's local observation with no knowledge from another agent. 

In MADDPG, all agents can have different or the same reward structures allowing for both cooperative (as in this Tennis example) and competitive agent behavior.

### DDPG Explanation (from previous project)

Actor-Critic methods take advantage of the strengths of both policy-based methods and value-based methods. By combining both kinds of methods, the negatives (for policy-based methods, high variance; for value-based methods, biased results) are reduced, hopefully giving an overall better agent with quicker convergence.

The Actor in DDPG attempts to predict the best action for every given state (a deterministic policy, instead of a stocastic policy from other Actors in Actor-Critic methods). The critic then uses the actors best action, along with the given state, to estimate the total value of that action in that state. This is very similar to the TD update used in DQN (see project 1).

DDPG also takes advantage of a Replay Buffer.

Like DQN, DDPG maintains two copies of the neural network, the "regular" network and the "target" network. DQN runs a bunch of time steps then copies the weights learned in the regular network into the target network all at once. In DDPG, this process is done more gradually in the form of a "Soft-Update". The TAU parameter controls how much the target network is moved towards the regular network at each step.

## Plot of Rewards

![Performance Chart](performance.png)

After hitting running 2686 games of Tennis (episodes), the past 100 episodes met the +0.5 score requirement to consider the enviroment solved. This means that the agent's were able to keep a volley going where each agent hit the ball four times and one hit it five times on average over the past 100 games. In the performance chart, there's a maximum score around 2.6 - this is likely the maximum score or ear maximum score before the time limit is hit (1000) and the game is reset.

## Model Weights

Saved Model weights can be found in the file episode-2686.pt. It's a dictonary containing the four networks (each agent's actor and critic).

## Ideas for Future Work

Prioritized Replays (where "boring" experiences are dropped) would also help improve the agents and help train quicker. This especially true early on since the agent's are essentially randomly responding to their observations and slowly learning how not to hit the ball over the net (or miss it entirely). The first few "hits" will be very valuable, especially when comparred to the potentially hundreds of games of "misses".

A different noise reduction schedule would also help train quicker. Instead of reducing it at a constant rate by episode, reducing it based on the score would help the noise shrink sooner, once the agent's on a good path and able to reliably return the ball, and keeping it high when it can't.
