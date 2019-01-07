from gym import GoalEnv
from gym.spaces import Discrete, Dict, MultiDiscrete


class SimpleFetchMdp(GoalEnv):
    def __init__(self, x_dim=5, y_dim=5, **kwargs):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_states = x_dim * y_dim
        # Right, Up, Left, Down, Grab
        self.action_space = Discrete(5)
        self.observation_space = Dict(dict(
            desired_goal=Discrete(self.num_states),  # Goal Position
            achieved_goal=Discrete(self.num_states),  # block position
            observation=MultiDiscrete([self.num_states, 2])  #arm position, object in air
        ))

        self._location_space = Discrete(self.num_states)
        self._goal_location = self._location_space.sample()
        self._block_location = self._location_space.sample()
        self._arm_location = self._location_space.sample()
        self._picked_up_block = False

        self.action_handlers =  [
            self._move_function(lambda s: s - 1, lambda s: s % self.x_dim == 0),  # right
            self._move_function(lambda s: s - self.x_dim, lambda s: s < self.x_dim),  # up
            self._move_function(lambda s: s + 1, lambda s: (s + 1) % self.x_dim == 0),  # left
            self._move_function(lambda s: s + self.x_dim, lambda s: s + self.x_dim >= self.x_dim * self.y_dim),  # down
            self._grab
        ]

    def reset(self):
        # Pick a random goal and block location
        self._goal_location = self._location_space.sample()
        self._block_location = self._location_space.sample()
        while self._block_location == self._goal_location:
            self._block_location = self._location_space.sample()
        self._arm_location = self._location_space.sample()
        self._picked_up_block = False
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed="None"):
        pass

    def step(self, action):
        self.action_handlers[action]()
        obs = self._get_obs()
        reward = self.compute_reward()
        done = reward == 1.
        info = []
        return obs, reward, done, info

    # Shortcut for setting the state and getting the output of the action
    def step_for(self, state, action, obs_format='dict'):
        self._goal_location = state[0]
        self._arm_location = state[2]
        if state[1] == -1:
            self._block_location = self._arm_location
            self._picked_up_block = True
        else:
            self._block_location = state[1]
            self._picked_up_block = False

        result = self.step(action)
        if obs_format == 'dict':
            return result
        return ([
            result[0]['desired_goal'],
            result[0]['achieved_goal'],
            result[0]['observation']
        ], ) + result[1:]

    def compute_reward(self):
        if self._arm_location == self._goal_location and self._picked_up_block:
            return 1.
        return 0.

    def _get_obs(self):
         return dict(
            desired_goal=self._goal_location,
            achieved_goal= -1 if self._picked_up_block else self._block_location,
            observation=self._arm_location
        )

    def _grab(self):
        if self._arm_location == self._block_location:
            self._picked_up_block = True


    def _move_function(self, displace, no_move_if):
        def ret():
            if no_move_if(self._arm_location):
                return
            self._arm_location=displace(self._arm_location)
            if self._picked_up_block:
                self._block_location = displace(self._block_location)
        return ret