import gym
from gym.envs.registration import register
import numpy as np

register(
    id='FetchMDP-v0',
    entry_point='fetch_mdp:FetchMdp',
)

env = gym.make('FetchMDP-v0')


env.reset()


GRID_DIMS = [0.25, .35]
GRID_CENTER = [1.3, 0.75]
info = [1.3, 0.75, 0.2]

observation, reward, done, info = env.step(4, render=True)

def move_to_goal(goal, observation):
    for i in range(1000):
        env.render()
        actions = env.action_space
        # actions is a 4-tuple: (forward/backward, right-left, up-down, open-close)
        current = observation['observation']
        # TODO: Make the policy stochastic to avoid tiebreaking problems
        if goal == current:
            return
        if goal % 5 > current % 5:
            action = 2
        elif goal % 5 < current % 5:
            action = 0
        elif np.floor(goal / 5) > np.floor(current / 5):
            action = 3
        elif np.floor(goal / 5) < np.floor(current / 5):
            action = 1
        print(action)
        observation, reward, done, info = env.step(action, render=True) # take a random action
        print(observation)

move_to_goal(observation['achieved_goal'], observation)

action = 4
observation, reward, done, info = env.step(action, render=True)

move_to_goal(observation['desired_goal'], observation)

action = 4
observation, reward, done, info = env.step(action, render=True)

while True:
    env.render()