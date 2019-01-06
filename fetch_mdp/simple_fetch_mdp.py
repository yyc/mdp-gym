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

        self.action_displacements =  [
            (lambda s: s - 1),  # right
            (lambda s: s - self.x_dim),  # up
            (lambda s: s + 1),  # left
            (lambda s: s + self.x_dim),  # down
        ]
        self.action_handlers = [self._move_function(displace) for displace in self.action_displacements]
        self.action_handlers.append(self._grab)

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


    def compute_reward(self):
        if self._arm_location == self._goal_location and self._picked_up_block:
            return 1.
        return 0.

    def _get_obs(self):
         return dict(
            desired_goal=self._goal_location,
            achieved_goal=self._block_location,
            observation=[self._arm_location, self._picked_up_block]
        )

    def _grab(self):
        if self._arm_location == self._block_location:
            self._picked_up_block = True


    def _move_function(self, displace):
        def ret():
            self._arm_location=displace(self._arm_location)
            if self._picked_up_block:
                self._block_location = displace(self._block_location)
        return ret