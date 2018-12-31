from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.spaces import Discrete, Dict

import numpy as np

"""
Wraps the FetchPickAndPlace gym environment in an MDP, so there are only 5 possible actions
and x * y * 2 possible states 
"""

actions = [
    (0.,1.,0.,0.),
    (1.,0.,0.,0.),
    (0.,-1.,0.,0.),
    (-1.,0.,0.,0.)
]

# From https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py#L76
# actual displacement when the action is 1.
POS_MODIFIER = 0.05

# Our own movement modifier to smooth movement
MOV_MODIFIER = 0.2

# From gym/robotics/assets/fetc/pick_and_place.xml
# forward-backward, left-right
GRID_DIMS = [0.25, 0.35]
GRID_CENTER = [1.3, 0.75]

class FetchMdp(FetchPickAndPlaceEnv):

    def __init__(self, x_dim=5, y_dim=5, **kwargs):
        super(FetchMdp, self).__init__('sparse', **kwargs)
        self._old_action_space = self.action_space
        self._old_observation_space = self.observation_space
        self.action_space = Discrete(5) #Right, Up, Left, Down, Grab
        self.observation_space = Dict(dict(
            desired_goal=Discrete(x_dim * y_dim),
            achieved_goal=Discrete(x_dim * y_dim),
            observation=Dict(dict(
                position=Discrete(x_dim * y_dim),
                target_position=Discrete(x_dim * y_dim + 1)
            ))
        ))
        self.action_handlers = [self.__move_function(a) for a in actions]
        self.action_handlers.append(self.__grab)

    def step(self, action, render=False):
        response = self.action_handlers[action](render=render)
        obs = response[0]
        state = self._position_to_state(obs)
        return state, response[1], response[2], response[3]

    # the superclass action is a 4-tuple: (forward/backward, right-left, up-down, open-close)
    def __move_function(self, vector):
        vector = np.array(vector) * MOV_MODIFIER
        # function is returned and used to handle movement
        def ret(render):
            obs = self._get_obs()
            current_state = self._position_to_state(obs)
            for _ in range(2):
                if render:
                    super(FetchMdp, self).render()
                response = self._step(vector)
            return response
        return ret

    def __grab(self, render):
        open_claw = np.array((0.,0.,0.,1.))
        for _ in range(2):
            if render:
                super(FetchMdp, self).render()
            response = self._step(open_claw)
        return response

    # Same as in the parent class except removes the np.clip
    def _step(self, action):
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _position_to_state(self, observation):
        pos = position['observation'][:3]
        position['observation'] = pos
        return position