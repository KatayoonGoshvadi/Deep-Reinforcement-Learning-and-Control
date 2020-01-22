import matplotlib.pyplot as plt
import numpy as np
import gym
import envs
import os.path as osp
import torch
from agent import Agent, RandomPolicy
from mpc import MPC
from model import PENN
import pdb
import time
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
# POPSIZE = 1000
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8

LOG_DIR = './data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TIME = time.time()
class ExperimentGTDynamics(object):
    def __init__(self, env_name='Pushing2D-v1', mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        # Does not need model
        self.warmup = False
        mpc_params['use_gt_dynamics'] = True
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer='cem'):
        # optimizer = "random"
        samples = []
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success


class ExperimentModelDynamics:
    def __init__(self, env_name='Pushing2D-v1', num_nets=1, mpc_params=None, exp_dir=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON
        self.num_nets = num_nets
        self.agent = Agent(self.env)
        mpc_params['use_gt_dynamics'] = False

        self.model = PENN(num_nets, STATE_DIM, len(self.env.action_space.sample()), LR)
        
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)
        
        self.random_policy_no_mpc = RandomPolicy(len(self.env.action_space.sample()))
        if exp_dir:
            self.exp_dir = os.path.join(os.getcwd(),exp_dir)
        else:
            self.exp_dir = os.getcwd()

        self.create_dirs(self.exp_dir)

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        print("optimizer: {}".format(optimizer))
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_epochs):
        """ Train a single probabilistic model using a random policy """
        samples = []
        for i in range(num_episodes):
            samples.append(self.agent.sample(self.task_horizon, self.random_policy_no_mpc))
        epoch_loss_arr, epoch_rmse_arr =  self.cem_policy.train(
                                                                [sample["obs"] for sample in samples],
                                                                [sample["ac"] for sample in samples],
                                                                [sample["rewards"] for sample in samples],
                                                                epochs=num_epochs
                                                            )
        self.save_data(epoch_loss_arr,"loss_warmup",self.exp_dir)
        self.save_data(epoch_rmse_arr,"rmse_warmup",self.exp_dir)
        self.plot_prop(epoch_loss_arr,"loss_warmup",self.exp_dir)
        self.plot_prop(epoch_rmse_arr,"rmse_warmup",self.exp_dir)
    
    def train(self, num_train_epochs, num_episodes_per_epoch, evaluation_interval):
        """ Jointly training the model and the policy """
        random_SR = []
        cem_SR = []
        loss_arr = []
        rmse_arr = []
        for i in range(num_train_epochs):
            print("####################################################################")
            print("Starting training epoch %d." % (i + 1))

            samples = []
            for j in range(num_episodes_per_epoch):
                samples.append(
                    self.agent.sample(
                        self.task_horizon, self.cem_policy
                    )
                )
            print("Rewards obtained:", [sample["reward_sum"] for sample in samples])
            loss_arr_curr, rmse_arr_curr = self.cem_policy.train(
                                                                [sample["obs"] for sample in samples],
                                                                [sample["ac"] for sample in samples],
                                                                [sample["rewards"] for sample in samples],
                                                                epochs=5
                                                            )

            loss_arr.append(np.mean(loss_arr_curr))
            rmse_arr.append(np.mean(rmse_arr_curr))
            print("mean_loss: {}".format(np.mean(loss_arr_curr)))
            print("mean_rmse: {}".format(np.mean(rmse_arr_curr)))
            
            self.save_data(loss_arr,"loss",self.exp_dir)
            self.save_data(rmse_arr,"rmse",self.exp_dir)

            if (i + 1) % evaluation_interval == 0:
                avg_return, avg_success = self.test(20, optimizer='cem')
                cem_SR.append(avg_success*1.0)
                print('Test success CEM + MPC:', avg_success)
                avg_return, avg_success = self.test(20, optimizer='random')
                random_SR.append(avg_success*1.0)
                print('Test success Random + MPC:', avg_success)
                
                self.save_data(cem_SR,"SR_CEM",self.exp_dir)
                self.save_data(random_SR,"SR_random",self.exp_dir)
                self.plot_prop(cem_SR,"SR_CEM",self.exp_dir)
                self.plot_prop(random_SR,"SR_random",self.exp_dir)

        self.save_data(cem_SR,"SR_CEM",self.exp_dir)
        self.save_data(random_SR,"SR_random",self.exp_dir)
        
        self.plot_prop(cem_SR,"SR_CEM",self.exp_dir)
        self.plot_prop(random_SR,"SR_random",self.exp_dir)
        

    def plot_prop(self,prop,prop_name,plots_path):
        fig = plt.figure(figsize=(16, 9))
        plt.plot(prop,label=prop_name,color='navy')
        plt.xlabel("num_epochs / 50")
        plt.ylabel(prop_name)
        plt.legend()
        plt.savefig(os.path.join(plots_path,"{}.png".format(prop_name)))
        plt.clf()
        plt.close()

    def save_data(self,prop,prop_name,save_dir):
        np.save(os.path.join(save_dir,prop_name+".npy"),prop)


    def create_dirs(self,path):
        if not os.path.exists(path):
            os.mkdir(path)

def test_cem_gt_dynamics(num_episode=10):
    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('CEM PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    
    # mpc_params = {'use_mpc': True, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    
    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('CEM PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode, optimizer='random)
    # print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': True, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    # print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))


def train_single_dynamics(num_test_episode=50,optimizer='cem'):
    num_nets = 1
    num_episodes = 1000
    num_epochs = 100
    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params,exp_dir="single_dynamics")
    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)
    curr_time = time.time()
    warm_up_time = curr_time - START_TIME
    print("Time_taken_for_warmup: {}".format(warm_up_time))
    avg_reward, avg_success = exp.test(num_test_episode, optimizer=optimizer)
    test_time = time.time() - curr_time
    print("Time_for_test: {}".format(test_time))
    print("avg_reward: {} avg_success: {}".format(avg_reward,avg_success)) 

def train_pets():
    num_nets = 2
    num_epochs = 500
    evaluation_interval = 20
    num_episodes_per_epoch = 1

    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params,exp_dir="pets")
    exp.model_warmup(num_episodes=100, num_epochs=10)
    exp.train(num_train_epochs=num_epochs,
              num_episodes_per_epoch=num_episodes_per_epoch,
              evaluation_interval=evaluation_interval)


if __name__ == "__main__":
    # test_cem_gt_dynamics(50)
    # train_single_dynamics(50,optimizer='cem')
    train_pets()
