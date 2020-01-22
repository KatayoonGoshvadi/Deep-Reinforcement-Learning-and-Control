import gym
import numpy as np

MAX_STEPS = 50

class PointEnv(gym.Env):

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=np.zeros(2), high=np.ones(2))
        self.action_space = gym.spaces.Box(low=np.full(2, -0.1), high=np.full(2, 0.1))
        self.state = None

    def reset(self):
        self.state = np.full(2, 0.1)
        self.num_steps = 0
        return self.state.copy()

    def step(self, action):
        action = action.clip(-0.1, 0.1)
        self.state = np.clip(self.state + action, 0.0, 1.0)
        goal = np.full(2, 0.9)
        dist = np.linalg.norm(self.state - goal)
        if dist < 0.1:
            r = 0.0
            done = True
            info = {"done": "goal reached"}
        elif self.num_steps == MAX_STEPS:
            r = -1.0
            done = True
            info = {"done": "max_steps_reached"}
        else:
            r = -1.0
            done = False
            info = {"done": None}
        self.num_steps += 1
        return self.state.copy(), r, done, info

