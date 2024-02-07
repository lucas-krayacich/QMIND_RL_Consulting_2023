import gymnasium as gym
import random 
import time

from stable_baselines3 import DQN
from IPython.display import clear_output


env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.reset()

model = DQN("MlpPolicy", env, buffer_size=10000, verbose=1)
model.learn(total_timesteps=10)

for steps in range(200): 
    obs, info = env.reset()
    done = False
    
    while not done: 
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: 
            obs, info = env.reset()
            print("Just died on episode ", steps)

env.close()