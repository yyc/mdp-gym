import os
from gym.spaces import Discrete, Dict
from gym.envs.robotics import fetch_env

import numpy as np

"""
Wraps the FetchPickAndPlace gym environment in an MDP, so there are only 5 possible actions
and x * y * 2 possible states 
"""

# From https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py#L76
# actual displacement when the action is 1.
POS_MODIFIER = 0.05

# Our own movement modifier to smooth movement
MOV_MODIFIER = 0.2

# From gym/robotics/assets/fetc/pick_and_place.xml
# forward-backward, left-right
GRID_DIMS = [0.25, 0.35]
GRID_CENTER = [1.3, 0.75]

# file_path = os.path.dirname(os.path.realpath(__file__))
# MODEL_XML_PATH = os.path.join(file_path, 'fetch_mdp.xml')
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

class FetchMdp(fetch_env.FetchEnv):
    def __init__(self, x_dim=5, y_dim=5, **kwargs):

        # From https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.3, 0.75, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.10, target_range=0.10, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type="sparse")

        self.x_states = x_dim
        self.y_states = y_dim
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
        self.action_handlers = [
            self._move_function((0., 1., 0., -1.), no_move_if=lambda s: s % self.x_states == 0),
            self._move_function((1., 0., 0., -1.), no_move_if=lambda s: s < self.x_states ),
            self._move_function((0., -1., 0., -1.), no_move_if=lambda s: (s + 1) % self.x_states == 0),
            self._move_function((-1., 0., 0., -1.), no_move_if=lambda s: s / self.x_states > (self.y_states - 1))
]
        self.action_handlers.append(self._grab)

    def step(self, action, render=False):
        response = self.action_handlers[action](render=render)
        obs = response[0]
        state = self._process_observation(obs)
        return state, response[1], response[2], response[3]

    # the superclass action is a 4-tuple: (forward/backward, right-left, up-down, open-close)
    def _move_function(self, vector, no_move_if=(lambda x: False)):
        move_vector = np.array(vector) * MOV_MODIFIER

        # function is returned and used to handle movement
        def ret(render):
            obs = self._get_obs()
            current_state = self._position_to_state(obs['observation'][:3])
            # If it's not a valid movement, do nothing
            if no_move_if(current_state):
                response = self._step(np.array((0., 0., 0., 0.)))
                return response
            while self._position_to_state(obs['observation'][:3]) == current_state:
                if render:
                    super(FetchMdp, self).render()
                response = self._step(move_vector)
                obs = response[0]
            return response
        return ret

    def _grab(self, render):
        open_claw = np.array((0.,0.,0.,1.))
        close_claw = np.array((0., 0., 0., -1.))
        down = np.array((0., 0., -1., 0.))
        up = np.array((0., 0., 1., 0.,))
        movement = [open_claw, down, down, down, close_claw, close_claw, up, up, up]
        for action in movement:
            if render:
                super(FetchMdp, self).render()
            response = self._step(action)
            if(response[1] == 1.):
                return response
        return response

    # Same as in the parent class except removes the np.clip
    def _step(self, action):
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        pos = obs['observation']

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info) + 1.
        return obs, reward, done, pos

    def _position_to_state(self, position):
        row = np.floor(self.y_states * ((GRID_CENTER[0] + GRID_DIMS[0]/2) - position[0]) / GRID_DIMS[0])
        col = np.floor(self.x_states * ((GRID_CENTER[1] + GRID_DIMS[1]/2) - position[1]) / GRID_DIMS[1])
        print((col, row))
        return row * self.x_states + col
        return col, row

    def _process_observation(self, obs):
        obs['achieved_goal'] = self._position_to_state(obs['achieved_goal'])
        obs['desired_goal'] = self._position_to_state(obs['desired_goal'])
        obs['observation'] = self._position_to_state(obs['observation'][:3])
        return obs
