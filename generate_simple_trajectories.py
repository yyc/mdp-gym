import gym
from gym.envs.registration import register
import numpy as np

register(
    id='FetchMDP-v0',
    entry_point='fetch_mdp:SimpleFetchMdp',
)

env = gym.make('FetchMDP-v0')

GRIDWORLD_X = 5
GRIDWORLD_Y = 5

action_displacements = [
    (lambda s: s - 1),  # right
    (lambda s: s - GRIDWORLD_X),  # up
    (lambda s: s + 1),  # left
    (lambda s: s + GRIDWORLD_X),  # down
]


def record(traj, observation, action):
    stateaction = [
        observation['desired_goal'],
        observation['achieved_goal'],
        observation['observation'],
        action]
    traj.append(stateaction)


def compute_trajectory():
    observation = env.reset()
    traj = []

    def move_to_goal(pos, goal):
        while pos != goal:
            nonlocal observation
            # So ties between equally good choices are broken at random
            valid_choices = []
            x_coords = (pos % GRIDWORLD_X), (goal % GRIDWORLD_X)
            if x_coords[0] != x_coords[1]:
                valid_choices.append(2 if x_coords[0] < x_coords[1] else 0)
            y_coords = (np.floor(pos / GRIDWORLD_X)), (np.floor(goal / GRIDWORLD_X))
            if y_coords[0] != y_coords[1]:
                valid_choices.append(3 if y_coords[0] < y_coords[1] else 1)
            action = np.random.choice(valid_choices)

            record(traj, observation, action)
            observation, reward, done, info = env.step(action)
            pos = observation['observation']

    move_to_goal(observation['observation'], observation['achieved_goal'])
    record(traj, observation, 4)  # Grab
    observation, reward, done, info = env.step(4)
    move_to_goal(observation['observation'], observation['desired_goal'])
    record(traj, observation, None)  # Record final state with no action
    return traj


trajectories = [compute_trajectory() for _ in range(1000)]
np.save('1000_mdp_trajectories', np.array(trajectories))


state = (0, 0, 1)
action = 0
# You can also get the next state in the environment using:
observation, reward, done, info = env.step_for(state, action)
print(observation)
# Which gives the observation in the openai dict format or
observation, reward, done, info = env.step_for(state, action, obs_format='tuple')
print(observation) # goal position, block position, arm position