import Box2D
from Box2D.b2 import (circleShape, fixtureDef, polygonShape)
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

MIN_COORD = 0
MAX_COORD = 5
PUSHER_START = np.array([1.0, 1.0])
BOX_START = np.array([2.0, 2.0])
FORCE_MULT = 1
RAD = 0.2
SIDE_GAP_MULT = 2
BOX_RAD = 0.2
GOAL_RAD = 0.5
MAX_STEPS = 20
FPS = 2

class Pusher2d(gym.Env):
    """A 2D pusher environment.
    
    The agent controls a circular paddle which pushes a circlular puck.
    The aim is to push the puck to a specified goal location.
    """

    def __init__(self):
        """Initialize the pusher environment."""
        self.seed()
        self.world = Box2D.b2World(gravity=(0,0))
        self.pusher = None
        self.box = None
        #Actions: x-movement, y-movement (clipped -1 to 1)
        self.action_space = spaces.Box(np.ones(2) * -1, \
                                       np.ones(2), dtype=np.float32)
        #State: x-pusher, y-pusher, x-box, y-box, x-goal, y-goal
        self.observation_space = spaces.Box(np.ones(6) * MIN_COORD, \
                                            np.ones(6) * MAX_COORD, dtype=np.float32)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_place(self):
        """Randomly samples a goal location nearby the initial location."""
        return [ \
            self.np_random.uniform(BOX_START[0] + BOX_RAD + GOAL_RAD, 
                                   MAX_COORD - RAD*SIDE_GAP_MULT),\
            self.np_random.uniform(BOX_START[1] + BOX_RAD + GOAL_RAD, 
                                   MAX_COORD - RAD*SIDE_GAP_MULT)]

    def _destroy(self):
        """Removes the Box2D entities."""
        if not self.box:
            return
        self.world.DestroyBody(self.box)
        self.world.DestroyBody(self.pusher)

    def reset(self):
        """Resets the environment.

        Returns:
            obs: initial observation.
        """
        self._destroy()
        self.pusher = self.world.CreateDynamicBody(
            position = PUSHER_START[:],
            fixtures = fixtureDef(
                shape=circleShape(radius=RAD, pos=(0,0)),
                density=1.0
                )
        )
        self.box = self.world.CreateDynamicBody(
            position = BOX_START[:],
            fixtures = fixtureDef(
                shape=circleShape(radius=BOX_RAD, pos=(0,0)),
                density=1.0
                )
        )
        self.goal_pos = self.random_place()
        self.elapsed_steps = 0
        return self._get_obs()

    def step(self, action):
        """Steps the environment.

        Args:
            action: The action to apply in the environment.
        Returns:
            obs: (array) the state of the environment.
            rew: (float) the reward for taking this action.
            done: (bool) an indicator of whether the episode terminated.
            info: (dict) additional information for debugging.
        """
        action = np.clip(action, -1, 1).astype(np.float32)
        self.elapsed_steps += 1
        self.pusher._b2Body__SetLinearVelocity((FORCE_MULT*action[0], FORCE_MULT*action[1]))
        self.box._b2Body__SetActive(True)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        done = False
        reward = -1
        obj_coords = np.concatenate([self.pusher.position.tuple, self.box.position.tuple])
        info = {"done": None}
        # Terminate the episode if the pusher or block is too far away.
        if np.min(obj_coords) < MIN_COORD or np.max(obj_coords) > MAX_COORD:
            reward = -1.0 * MAX_STEPS
            done = True
            info['done'] = 'out of bounds'
        # Check if out of time.
        elif self.elapsed_steps >= MAX_STEPS:
            done = True
            info["done"] = "max steps reached"
        # Check if goal reached.
        elif np.linalg.norm(np.array(self.box.position.tuple) - self.goal_pos) < RAD + GOAL_RAD:
            done = True
            reward = 0
            info["done"] = "goal reached"
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Compute the observation.
        
        Coordinates [0, 1] are the pusher position.
        Coordinates [2, 3] are the puck position.
        Coordinates [4, 5] are the goal position for the puck.
        """
        state = np.concatenate([self.pusher.position.tuple, \
                                self.box.position.tuple, \
                                self.goal_pos])
        return state

    def apply_hindsight(self, states):
        """Relabels a trajectory using a new goal state.

        This involves modifying each state to correspond to the new goal,
        and recomputing the corresponding rewards.
        
        Args:
            states: (list) states in a trajectory.
        Returns:
            her_states: (list) states in a trajectory.
            her_rewards: (list) rewards for the relabeled trajectory.
        """
        goal = states[-1][2:4] # Get new goal location (last location of box).
        her_states = []
        her_rewards = []
        for s in states:
            s[-2:] = goal.copy()
            r = self._HER_calc_reward(s)
            her_states.append(s)
            her_rewards.append(r)
        return her_states, her_rewards

    def _HER_calc_reward(self, state):
        """Computes the reward for a given state, which contains the goal.

        Args:
            state: a state, part of which corresponds to the goal.
        Returns
            reward: (float) the reward.
        """
        if np.linalg.norm(state[2:4] - state[4:6]) < RAD + GOAL_RAD:
            return 0.0
        else:
            return -1.0
