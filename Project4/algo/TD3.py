import numpy as np
import matplotlib.pyplot as plt
import os
from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork
import torch
import pdb
import matplotlib as mpl
mpl.use('Agg')
BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.99                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class EpsilonNormalActionNoise(object):
        """A class for adding noise to the actions for exploration."""

        def __init__(self, mu, sigma, epsilon):
                """Initialize the class.

                Args:
                        mu: (float) mean of the noise (probably 0).
                        sigma: (float) std dev of the noise.
                        epsilon: (float) probability in range [0, 1] with
                        which to add noise.
                """
                self.mu = [mu,mu]
                self.sigma = [sigma,sigma]
                self.epsilon = epsilon

        def __call__(self, action):
                """With probability epsilon, adds random noise to the action.
                Args:
                        action: a batched tensor storing the action.
                Returns:
                        noisy_action: a batched tensor storing the action.
                """
                if np.random.uniform() > self.epsilon:
                        return action + np.random.normal(self.mu, self.sigma)
                else:
                        return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG_TD3(object):
        """A class for running the DDPG algorithm."""

        def __init__(self, env, lr_a,lr_c, sigma,data_path,plots_path, outfile_name,device,delay,logger=None):
                """Initialize the DDPG object.

                Args:
                        env: an instance of gym.Env on which we aim to learn a policy.
                        outfile_name: (str) name of the output filename.
                """
                action_dim = len(env.action_space.low)
                state_dim = len(env.observation_space.low)
                self.sigma = sigma
                self.outfile = outfile_name
                self.data_path = data_path
                self.plots_path = plots_path
                self.device = device
                self.logger = logger
                self.delay = delay
                self.actor = ActorNetwork(state_size=state_dim, 
                                            action_size=action_dim, 
                                            learning_rate=lr_a,
                                            device=device)

                self.critic_1 = CriticNetwork(state_size=state_dim,
                                            action_size=action_dim, 
                                            learning_rate=lr_c,
                                            device=device)

                self.critic_2 = CriticNetwork(state_size=state_dim,
                                            action_size=action_dim, 
                                            learning_rate=lr_c,
                                            device=device)

                self.actor_target = ActorNetwork(state_size=state_dim, 
                                                    action_size=action_dim, 
                                                    learning_rate=lr_a,
                                                    device=device)

                self.critic_1_target = CriticNetwork(state_size=state_dim, 
                                                    action_size=action_dim, 
                                                    learning_rate=lr_c,
                                                    device=device) 

                self.critic_2_target = CriticNetwork(state_size=state_dim, 
                                                    action_size=action_dim, 
                                                    learning_rate=lr_c,
                                                    device=device) 

                if(device.type=="cuda"):
                        self.actor.to(device)
                        self.critic_1.to(device)
                        self.critic_2.to(device)
                        self.actor_target.to(device)
                        self.critic_1_target.to(device)
                        self.critic_2_target.to(device)
                        print("models moved to gpu")
                else:
                    print("models on cpu")



                self.replay_buff = ReplayBuffer(BUFFER_SIZE)

                np.random.seed(1337)
                self.env = env

        def evaluate(self, num_episodes):
                """Evaluate the policy. Noise is not added during evaluation.

                Args:
                        num_episodes: (int) number of evaluation episodes.
                Returns:
                        success_rate: (float) fraction of episodes that were successful.
                        average_return: (float) Average cumulative return.
                """
                test_rewards = []
                success_vec = []

                self.actor.eval()

                fig = plt.figure(figsize=(12, 12))
                for i in range(num_episodes):
                        s_vec = []
                        state = self.env.reset()
                        s_t = np.array(state)
                        total_reward = 0.0
                        done = False
                        step = 0
                        success = False
                        while not done:
                                s_vec.append(s_t)
                                with torch.no_grad():
                                        a_t = self.actor(torch.tensor(s_t))
                                new_s, r_t, done, info = self.env.step(a_t.cpu().numpy())
                                if done and "goal" in info["done"]:
                                        success = True
                                new_s = np.array(new_s)
                                total_reward += r_t
                                s_t = new_s
                                step += 1
                        success_vec.append(success)
                        test_rewards.append(total_reward)
                        if i < 9:
                                plt.subplot(3, 3, i+1)
                                s_vec = np.array(s_vec)
                                pusher_vec = s_vec[:, :2]
                                puck_vec = s_vec[:, 2:4]
                                goal_vec = s_vec[:, 4:]
                                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                                                 color='g' if success else 'r')
                                plt.xlim([-1, 6])
                                plt.ylim([-1, 6])
                                if i == 0:
                                        plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                                if i == 8:
                                        plt.savefig(os.path.join(self.plots_path,"res.png"))
                                        plt.close()
                                        # plt.show()
                self.actor.train()
                return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

        def copyWeights(self,target_network,network,tau=TAU):
                for target_param, param in zip(target_network.parameters(), network.parameters()):
                        target_param.data.copy_(tau*param.data + target_param.data*(1.0 - tau))


        def train(self, num_episodes, hindsight=False):
                """Runs the DDPG algorithm.

                Args:
                        num_episodes: (int) Number of training episodes.
                        hindsight: (bool) Whether to use HER.
                """
                critic_loss_arr = []
                td_error_arr = []
                estimated_q_val_ep_arr = []
                q_first_state_arr = []
                actor_loss_arr  = []
                mean_test_rewards_arr = []
                std_test_rewards_arr = []
                self.actor.train()
                self.critic_1.train()
                self.critic_2.train()
                self.actor_target.train()
                self.critic_1_target.train()
                self.critic_2_target.train()

                
                self.copyWeights(self.actor_target,self.actor,tau=1.0)
                self.copyWeights(self.critic_1_target,self.critic_1,tau=1.0)
                self.copyWeights(self.critic_2_target,self.critic_2,tau=1.0)

                epsilon = 1.0
                for i in range(num_episodes):
                        epsilon = max((0.5 - 0.45*i/20000),0.05)
                        state = self.env.reset()
                        total_reward = 0.0
                        done = False
                        step = 0
                        loss = 0
                        actor_loss = 0
                        critic_loss = 0
                        estimated_q_before = 0
                        estimated_q_after = 0
                        store_states = []
                        store_actions = []
                        while not done:
                            step += 1
                            # Collect one episode of experience, saving the states and actions
                            # to store_states and store_actions, respectively.
                            with torch.no_grad():
                                action_mu = self.actor(torch.from_numpy(state))

                            AddNoise = EpsilonNormalActionNoise(0, self.sigma, epsilon)
                            action = AddNoise(action_mu.cpu().numpy())
                            with torch.no_grad():
                                state_tmp = torch.unsqueeze(torch.from_numpy(state),dim=0)
                                action_tmp = torch.unsqueeze(torch.from_numpy(action),dim=0)
                                estimated_q_before += self.critic_1(state_tmp,action_tmp).item()
                                if step==1:
                                    q_first_state_arr.append(estimated_q_before*1.0)
                            store_actions.append(action*1.0)
                            store_states.append(state*1.0)
                            next_state , reward, done, info = self.env.step(action)
                            total_reward = total_reward + reward
                            self.replay_buff.add(state, action, reward, next_state, done)
                            state = next_state.copy()

                            if hindsight == False:
                                if(self.replay_buff.count() >= 3*BATCH_SIZE):
                                    c_loss, a_loss, tde = self.update_networks(i)
                                    critic_loss += c_loss
                                    actor_loss += a_loss
                                    loss += tde

                                                
                        if hindsight:
                                # For HER, we also want to save the final next_state.
                                new_s = state.copy()
                                store_states.append(new_s)

                                self.add_hindsight_replay_experience(store_states, store_actions)
                                for _ in range(len(store_states)-1):                               
                                    if(self.replay_buff.count() >= 3*BATCH_SIZE):
                                        c_loss, a_loss, tde = self.update_networks()
                                        critic_loss += c_loss
                                        actor_loss += a_loss
                                        loss += tde

                        td_error_arr.append(loss/step)
                        actor_loss_arr.append(actor_loss/step)
                        
                        if not hindsight:
                            if(self.replay_buff.count() >= BATCH_SIZE):
                                with torch.no_grad():
                                    Batch = np.array(self.replay_buff.get_batch(BATCH_SIZE))
                                    sampled_states = torch.from_numpy(np.stack(Batch[:,0]))
                                    estimated_q_val_ep_arr.append(self.critic_1(sampled_states,self.actor(sampled_states)).mean().item())

    #                    self.add_logs({'estimated_q_val_ep':estimated_q_val_ep_arr[-1],
     #                                           'q_first_state':q_first_state_arr[-1],
      #                                          'td_error':td_error_arr[-1]
       #                                         },i)


                        del store_states, store_actions
                        store_states, store_actions = [], []
                        # Logging
                        print("Episode %d: Total reward = %d" % (i, total_reward))
                        print("\tTD loss = %.2f" % (loss / step,))
                        print("\tSteps = %d; Info = %s" % (step, info['done']))
                        if i % 100 == 0:
                                successes, mean_rewards, std = self.evaluate(10)

                                mean_test_rewards_arr.append(mean_rewards)
                                std_test_rewards_arr.append(std)

                                np.save(os.path.join(self.data_path,"mean_test_reward.npy"),np.array(mean_test_rewards_arr))
                                np.save(os.path.join(self.data_path,"std_test_reward.npy"),np.array(std_test_rewards_arr)) 
                                np.save(os.path.join(self.data_path,"actor_loss.npy"),np.array(actor_loss_arr))
                                np.save(os.path.join(self.data_path,"td_error.npy"),np.array(td_error_arr))
                                if not hindsight:
                                    np.save(os.path.join(self.data_path,"q_first_state.npy"),np.array(q_first_state_arr))
                                    np.save(os.path.join(self.data_path,"estimated_q_val_ep"),np.array(estimated_q_val_ep_arr))
                                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                                with open(self.outfile, "a") as f:
                                        f.write("%.2f, %.2f,\n" % (successes, mean_rewards))
          #                      self.add_logs({'mean_test_reward':mean_rewards,
           #                                     'actor_loss':actor_loss
            #                                    },i/100)

                self.plot_prop(td_error_arr,"td_error")
                if not hindsight:
                    self.plot_rewards(mean_test_rewards_arr,std_test_rewards_arr)
 #                   self.plot_prop(estimated_q_after_arr,"expected_return")
#                    self.plot_prop(estimated_q_before_arr,"estimated_q_before")
                    self.plot_prop(q_first_state_arr,"q_first_state_action")
                    self.plot_prop(estimated_q_val_ep_arr,"estimated_q_val_ep")

        def calc_q_vals_before_update(self,states,actions):
            with torch.no_grad(): 
                s_tensor = torch.tensor(states)
                a_tensor = torch.tensor(actions)
                estimated_q = self.critic(s_tensor,a_tensor).mean()
            return estimated_q.item()


        def calc_q_vals_after_update(self,states):
            with torch.no_grad(): 
                s_tensor = torch.tensor(states)
                a_tensor = self.actor(s_tensor)
                estimated_q = self.critic(s_tensor,a_tensor).mean()
            return estimated_q.item()

                        
        def add_hindsight_replay_experience(self, states, actions):
                """Relabels a trajectory using HER.

                Args:
                        states: a list of states.
                        actions: a list of states.
                """
                new_goal = states[-1][2:4].copy()
                num_dones = 0
                for state_id in range(len(states)-1):
                    ns = states[state_id+1].copy()
                    if np.sum(abs(new_goal-ns[2:4])) != 0:
                        num_dones = 0
                        s = states[state_id].copy()
                        a = actions[state_id].copy()
                        s[-2:] = new_goal.copy()
                        r = self.env._HER_calc_reward(s)
                        ns[-2:] = new_goal.copy()
                        done = False
                        if(np.linalg.norm(np.array(ns[2:4] - new_goal) < 0.7)):
                            done = True
                            num_dones += 1
                        self.replay_buff.add(s, a, r, ns, done)
                       # print("dones: {}".format(num_dones))

                # for num_state in range(len(states)-1):
                #     num_new_goals = 4
                #     # selecting num_new_goals number of new goals from the future puck positions of the episode
                #     # new_goal_idxs = np.random.randint(low=num_state, high=len(states), size=num_new_goals)

                #     # num_dones = 0
                #     for num_goal,new_goal_id in enumerate(new_goal_idxs): # number of goals
                #         new_goal = states[new_goal_id][2:4]
                #         s = states[num_state]
                #         a = actions[num_state]
                #         s[-2:] = new_goal.copy()
                #         r = self.env._HER_calc_reward(s)
                #         ns = states[num_state+1]
                #         ns[-2:] = new_goal.copy()
                #         done = False
                #         if(np.linalg.norm(np.array(ns[2:4] - new_goal) < 0.7)):
                #             done = True
                #             num_dones += 1
                #         self.replay_buff.add(s, a, r, ns, done)
                #     print("dones: {}".format(num_dones))

        def update_networks(self,i):

                Batch = np.array(self.replay_buff.get_batch(BATCH_SIZE))
                s = torch.from_numpy(np.stack(Batch[:,0]))
                a = torch.from_numpy(np.stack(Batch[:,1]))
                r = torch.tensor(np.stack(Batch[:,2]),dtype=torch.float32).to(device=self.device)
                ns = torch.from_numpy(np.stack(Batch[:,3]))
                d = torch.tensor(np.stack(Batch[:,4]),dtype=torch.float32).to(device=self.device)
                Q_pred_1 = self.critic_1(s,a)
                Q_pred_2 = self.critic_2(s,a)
                with torch.no_grad():
                        na = self.actor_target(ns)
                        q_val_nxt_1 = self.critic_1_target(ns,na)
                        q_val_nxt_2 = self.critic_2_target(ns,na)
                        Q_target = torch.unsqueeze(r,dim=1) +  GAMMA * torch.min(q_val_nxt_1,q_val_nxt_2) * torch.unsqueeze(1.0-d, dim=1)
            
                val_loss_1 = self.critic_1.mse_loss(Q_pred_1,Q_target)
                val_loss_2 = self.critic_2.mse_loss(Q_pred_2,Q_target)

                self.critic_1.optimizer.zero_grad()
                val_loss_1.backward()
                self.critic_1.optimizer.step()

                self.critic_2.optimizer.zero_grad()
                val_loss_2.backward()
                self.critic_2.optimizer.step()
                
                td_error_1 = torch.abs(Q_pred_1 - Q_target).mean().item()
                td_error_2 = torch.abs(Q_pred_2 - Q_target).mean().item()
                
                policy_loss = torch.tensor([0])
                if(i%self.delay):
                    self.critic_1.optimizer.zero_grad()
                    self.actor.optimizer.zero_grad()
                    policy_loss = -self.critic_1(s,self.actor(s)).mean()
                    
                    policy_loss.backward()
                    self.actor.optimizer.step()

                    self.copyWeights(self.actor_target,self.actor)
                    self.copyWeights(self.critic_1_target,self.critic_1)
                    self.copyWeights(self.critic_2_target,self.critic_2)

                return val_loss_1.item(),policy_loss.item(),td_error_1

        def plot_rewards(self,mean_arr,std_arr):
                mean = np.array(mean_arr)
                std = np.array(std_arr)
                fig = plt.figure(figsize=(16, 9))
                x = np.arange(0,mean.shape[0])
                plt.plot(x,mean, label="mean_cummulative_test_reward",color='orangered')
                plt.fill_between(x,mean-std, mean+std,facecolor='peachpuff',alpha=0.5)
                plt.xlabel("num episodes / {}".format(100))
                plt.ylabel("test_reward")
                plt.legend()
                plt.savefig(os.path.join(self.plots_path,"test_rewards.png"))
                plt.clf()
                plt.close()

        def plot_prop(self,prop,prop_name):
                fig = plt.figure(figsize=(16, 9))
                plt.plot(prop,label=prop_name,color='navy')
                plt.xlabel("num episodes")
                plt.ylabel(prop_name)
                plt.legend()
                plt.savefig(os.path.join(self.plots_path,"{}.png".format(prop_name)))
                plt.clf()
                plt.close()

        def add_logs(self,log_dict,n_iter):
            for key in log_dict.keys():
                self.logger.add_scalar(key,log_dict[key],n_iter)
