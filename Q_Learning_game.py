
# This is a sandbox notebook 
import pandas as pd
import numpy as np 
import gymnasium as gym
import random
import time 
from IPython.display import clear_output


env = gym.make("FrozenLake-v1", render_mode="human")
# env.reset()

# #Check how many states there are and how many actions per state (in this case there will always be the same number of actions per state)
action_space_size = env.action_space.n
state_space_size = env.observation_space.n



print("Number of Actions Per State: ")
print(action_space_size)
print("Number of States ")
print(state_space_size)

# Create a Q-table
# As the agent explores it will use a formula to update the values within the table to find out which state/actions result in the greatest reward

q_table = np.zeros((state_space_size, action_space_size))
# q_table
print(q_table)




# How many times the agent will play the game
num_episodes = 10
# If the agent uses 100 actions and hasnt won or lost then the game counts as a loss
max_steps_per_episode = 100

# Remember: Every time you hit a state you calculate a new q-value, the learning rate is essentially how much you want to change the q-value each time
learning_rate = 0.1
# discount rate is just (1 - learning_rate)
discount_rate = 0.99

# Exploration rate is the likelihood that the agent looks for new states rather than using the q-table values it already has
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
# Every iteration the exploratoin rate decreases a little bit that way the agent uses the q-table once it starts to get full
exploration_decay_rate = 0.001


rewards_all_episodes = []

#Q-learning algorithm: 
for episode in range(num_episodes):
    # new game is starting reset environment 
    state = env.reset()

    done = False
    rewards_current_episode = 0
    
    # this is the 'actions' loop, this loop chooses the action for the agent to do
    for step in range(max_steps_per_episode):
        #Chooses a random number between 0 and 1
        exploration_rate_threshold = random.uniform(0,1)
        # If this random number is bigger than the exploration rate then we wont 'explore' this step we will just make the agent use a value from the q-table
        if exploration_rate_threshold > exploration_rate: 
            # choose the action with the highest reward in the q_table
            try:
                action = np.argmax(q_table[state,:])
            except: 
                action = np.argmax(q_table[state[0],:])
        else: 
            # try a random action
            action = env.action_space.sample()
            print(action)
            print(state)

        new_state, reward, done, info_bool, info = env.step(action)
        env.render()

        # Update q-table for this action/state pair
        try: 
             q_table[state, action] = q_table[state, action]*(1-learning_rate) + \
            learning_rate*(reward + discount_rate * np.max(q_table[new_state,:]))
        except:
            q_table[state[0], action] = q_table[state[0], action]*(1-learning_rate) + \
                learning_rate*(reward + discount_rate * np.max(q_table[new_state,:]))
    
        #for some reason the state and action numbers go out of bounds of the qtable
        # this means there is an issue with the state and actions or with the creation of the q-table itself
        state = new_state
        rewards_current_episode += reward

        if done==True: 
            break

    # Exploration Rate decay
    # After every episode we reduce the exploration rate

    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    rewards_all_episodes.append(rewards_current_episode)

print("############################DONE##########################")
print("Updated Q-table")
print(q_table)
# rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
# count = 1000
# print("******************Average reward per thousand episodes**************\n")
# for r in rewards_per_thousand_episodes: 
#     print(count, ": ", str(sum(r/1000)))
#     count += 1000

# print("\n\n********************Q-table********************\n")
# print(q_table)




