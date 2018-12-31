import gym
from gym.envs.registration import register

register(
    id='FetchMDP-v0',
    entry_point='fetch_mdp:FetchMdp',
)

env = gym.make('FetchMDP-v0')


env.reset()


GRID_DIMS = [0.25, 0.35]
GRID_CENTER = [1.3, 0.75]
obs = [1.3, 0.75, 0.2]

for i in range(1000):
    env.render()
    actions = env.action_space
    #action = actions.sample()
    if obs[1] < (0.75 + 0.175):
        action = 0
    elif obs[0] < (1.3 + 0.125):
        action = 1
    else:
        action = 2
    # actions is a 4-tuple: (forward/backward, right-left, up-down, open-close)
    print(action)
    print(obs)
    observation, reward, done, info = env.step(action, render=True) # take a random action
    obs = observation['observation']
