# Import necessary packages
import gym
import random
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Training parameters for Q-learning
alpha = 0.9  # Learning rate
gamma = 0.9  # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500  # per each episode
avg_tot_reward = 0
avg_tot_actions = 0

# Q table for rewards
Q_reward = np.zeros((env.observation_space.n, env.action_space.n))

# Training
for episode in range(num_of_episodes):
    state = env.reset()[0]
    total_reward = 0
    for step in range(num_of_steps):
        action = np.random.randint(0,6)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            Q_reward[state,action] = reward
            break
        else:
            Q_reward[state, action] = Q_reward[state, action] + alpha * (reward + gamma * np.max(Q_reward[next_state]) - Q_reward[state, action])
            state=next_state


# Run the testing part ten times
for i in range(10):
    state = env.reset()[0]
    tot_reward = 0
    tot_actions = 0
    for t in range(50):
        action = np.argmax(Q_reward[state,:])
        state, reward, done, truncated, info = env.step(action)
        tot_reward += reward
        tot_actions += 1
        print(env.render())
        if done:
            print("Total reward %d" %tot_reward)
            print("Total number of actions %d" %tot_actions)
            break
    
    # Update average values
    avg_tot_reward += tot_reward
    avg_tot_actions += tot_actions

# Compute the averages
avg_tot_reward = avg_tot_reward/10
avg_tot_actions = avg_tot_actions/ 10

print(f"Average Total Reward: {avg_tot_reward}")
print(f"Average Number of Actions: {avg_tot_actions}")