import numpy as np
import pdb

class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False
        policy.reset()
        # policy.set_goal(states[0])
        for t in range(horizon):
            # print('time step: {}/{}'.format(t, horizon))
            # print("t: {}".format(t))
            actions.append(policy.act(states[t], t))
            state, reward, done, info = self.env.step(actions[t])
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            if done:
                # print(info['done'])
                break

        # print("Rollout length: ", len(actions))

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def reset(self):
        pass

    def act(self, arg1, arg2):
        return np.random.uniform(size=self.action_dim) * 2 - 1
