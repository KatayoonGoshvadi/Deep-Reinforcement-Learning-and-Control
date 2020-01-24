#!/usr/bin/env python
import numpy as np
import gym
import sys
import copy
import argparse
from collections import deque
import os
import random
import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
#import seaborn as sns
#sns.set()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 
	def __init__(self, environment_name,obs_space,action_space,lr):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		#pdb.set_trace()
		super().__init__()

		self.environment_name = environment_name
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.lr = lr
		self.input_dim = obs_space.shape[0]

		if (environment_name=="CartPole-v0"):
			self.linear1 = nn.Linear(self.input_dim,32)
			self.output_layer = nn.Linear(32,action_space.n)

		else:

			self.linear1 = nn.Linear(self.input_dim,64)
			self.linear2 = nn.Linear(64,64)
			self.linear3 = nn.Linear(64,64)
			self.output_layer = nn.Linear(64,action_space.n)

		self.initialize_parameters()

		self.create_optimizer()
		
	def forward(self,X):
		#pdb.set_trace()
		X = X.float().to(device=DEVICE)
		if (self.environment_name=="CartPole-v0"):
			x_em = F.relu(self.linear1(X))

		else:
			x_em = F.relu(self.linear1(X))
			x_em = torch.tanh(self.linear2(x_em))
			x_em = torch.tanh(self.linear3(x_em))

		
		out = self.output_layer(x_em)

		return out
	
	def create_optimizer(self,lr=None):
		if(not lr):
			lr = self.lr
		self.optimizer = torch.optim.Adam(self.parameters(),lr)

	def initialize_parameters(self):
		for param in self.parameters():
			if(len(param.shape)>1):
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.constant_(param, 0.0)

	def save_checkpoint(self, checkpoint_save_path, num_episodes_trained):
		# Helper function to save your model / weights. 
		torch.save({'model_state_dict': self.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'num_episodes_trained': num_episodes_trained}, checkpoint_save_path)
		print("checkpoint_saved")


	def load_checkpoint(self,checkpoint_path):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		#self.model.load_weights(model_file)
		# weight_file: full path of the pth file
		checkpoint = torch.load(checkpoint_path)
		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		num_episodes_trained = checkpoint['num_episodes_trained']
		
		return num_episodes_trained
		



class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		# Hint: you might find this useful:
		# 		collections.deque(maxlen=memory_size)
		
		
		self.replay_deque = deque(maxlen=memory_size)
		self.memory_size = memory_size
		self.burn_in = burn_in ## just for reference purposes
	

	def sample_batch(self, batch_size=32):
		
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		
		sampled_ids = np.random.choice(np.arange(len(self.replay_deque)),size=batch_size)

		sampled_transitions = [self.replay_deque[id] for id in sampled_ids]

		return np.array(sampled_transitions)

	def append(self, transition):
		# Appends transition to the memory. 
	
		self.replay_deque.append(transition)
		 


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, Q_model, DQ_model, render=False, learning_rate=1e-5, gamma=0.99, replay_size=100000, burn_in=20000, save_weights_path=None):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
	
		self.environment_name = environment_name
	
		# env for generating and storing transistions
		self.env = gym.make(self.environment_name)
		
		# test env
		self.test_env = gym.make(self.environment_name)
		
		self.obs_space = self.env.observation_space
		
		self.action_space = self.env.action_space
		self.nb_actions = self.action_space.n
		
		# creating the Q network		
		self.Q_net = Q_model
		self.DQ_net = DQ_model
		#pdb.set_trace()	
		self.DQ_net.load_state_dict(self.Q_net.state_dict())
		self.DQ_net.create_optimizer()

		self.replay_buffer = Replay_Memory(memory_size=replay_size,burn_in=burn_in)
		self.burn_in_memory(burn_in)
		self.gamma = gamma
		self.batch_size = 32
		

	def epsilon_greedy_policy(self, q_values, epsilon):
		# Creating epsilon greedy probabilities to sample from. 
	

		# go_greedy = np.random.choice(2,size=1,p=[epsilon,1-epsilon])[0]
		
		go_greedy = random.random()

		if(go_greedy > epsilon):
			action = np.argmax(q_values)
		else:
			action = np.random.choice(q_values.shape[1],size=1)[0]

		return action

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		
		action = np.argmax(q_values)
		return action


	def action_to_one_hot(env, action):
		action_vec = np.zeros(env.action_space.n)
		action_vec[action] = 1
		return action_vec 

	def custom_mse_loss(self, Y_pred, Y_target, actions):
		loss = 0
		for i in range(len(actions)):
			loss += torch.pow(Y_pred[i,actions[i]] - Y_target[actions[i]], 2)
		
		loss /= len(actions)

		# loss = torch.tensor([Y_pred[i,actions[i]] - Y_target[i,actions[i]] for i in range(0,len(actions))])
		# loss = torch.mean(torch.pow(loss,2))
		return loss

	def create_action_mask(self,actions):
		action_mask = np.zeros((actions.shape[0],self.nb_actions))
		for id, mask in enumerate(action_mask):
			mask[actions[id]] = 1
		return action_mask

	
	def train(self,epsilon,update_target=False):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		


		curr_state = self.env.reset() ## getting the first state 
		done = False
		

		while not(done):

			
			with torch.no_grad():
				q_values = self.Q_net(torch.unsqueeze(torch.from_numpy(curr_state),dim=0))
			
			action = self.epsilon_greedy_policy(q_values.cpu().numpy(), epsilon)
			
			
			next_state, reward, done, info = self.env.step(action) 	
			self.replay_buffer.append((curr_state,action,reward,next_state,done))
			curr_state = next_state.copy()

			## sample a minibatch of random transitions from the replay buffer
			sampled_transitions = self.replay_buffer.sample_batch(batch_size=self.batch_size)
			q_values_target = [None]*self.batch_size

			X_train = [None]*self.batch_size

			transition_actions = [None]*self.batch_size

			X_train = np.array([transition[0] for transition in sampled_transitions])
			transition_actions = np.array([transition[1] for transition in sampled_transitions])
			action_mask = torch.tensor(self.create_action_mask(transition_actions),dtype=torch.bool).to(device=DEVICE)
			exp_rewards = torch.tensor([transition[2] for transition in sampled_transitions]).float().to(device=DEVICE)
			sampled_nxt_states = np.array([transition[3] for transition in sampled_transitions])
			dones = np.array([int(transition[4]) for transition in sampled_transitions])

			with torch.no_grad():
				q_max_nxt_state,_ = torch.max(self.DQ_net(torch.from_numpy(sampled_nxt_states)),axis=1) 

			q_values_target = exp_rewards + self.gamma * q_max_nxt_state * torch.tensor(1-dones).float().to(device=DEVICE)
		
			Y_pred_all_actions = self.Q_net(torch.from_numpy(X_train))

			Y_pred = torch.masked_select(Y_pred_all_actions,action_mask)

			batch_loss = F.mse_loss(Y_pred,q_values_target)
			self.Q_net.optimizer.zero_grad()
			batch_loss.backward()
			self.Q_net.optimizer.step()

			if(update_target):
				self.DQ_net.load_state_dict(self.Q_net.state_dict())
				#print("target Q_net updated")

			#print("train_loss: {}".format(batch_loss.item()))
			
		return batch_loss.item()

        def copy_network_weights(self,net1,target_net):
            target_net.load_state_dict(net.state_dict())

	def test(self, test_num_episodes=100):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		# load the model weights if provided with a saved model	
	
		self.Q_net.eval()
		done = False
		episodic_test_rewards = []
		for episode in range(test_num_episodes):
			episode_reward = 0
			state_t = self.env.reset()
			done = False
			while not done:
				with torch.no_grad():
					q_values = self.Q_net(torch.unsqueeze(torch.from_numpy(state_t),dim=0))
				action = self.greedy_policy(q_values.cpu().numpy())
				state_t_1, reward, done, info = self.env.step(action)
				episode_reward += reward
				state_t = state_t_1.copy()
			episodic_test_rewards.append(episode_reward)
		self.Q_net.train()
	
		return np.mean(episodic_test_rewards), np.std(episodic_test_rewards)
		


	def burn_in_memory(self,burn_in):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		print("burn_in_start")

		nb_actions = self.action_space.n

		for i in range(burn_in):
			curr_state = self.env.reset()	
			done = False
			while not done:
				action = np.random.randint(0,nb_actions)
				next_state,reward,done,info = self.env.step(action)
				self.replay_buffer.append((curr_state,action,reward,next_state,done))
				curr_state = next_state.copy()

		print("burn_in_over")
		pass


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
	#       	test_video(self, self.environment_name, episode)
	save_path = "./videos-%s-%s" % (env, epi)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	# To create video
	env = gym.wrappers.Monitor(agent.env, save_path, force=True)
	reward_total = []
	state = env.reset()
	done = False
	while not done:
		env.render()
		action = agent.epsilon_greedy_policy(state, 0.05)
		next_state, reward, done, info = env.step(action)
		state = next_state
		reward_total.append(reward)
	print("reward_total: {}".format(np.sum(reward_total)))
	agent.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument("--lr",dest="lr",type=float,default=1e-5)
	parser.add_argument("--num-episodes",dest="num_episodes",type=int,default=1000)
	parser.add_argument("--test-after",dest="test_after",type=int,default=100)
	parser.add_argument("--gamma",dest="gamma",type=float,default=0.99)
	parser.add_argument("--replay-size",dest="replay_size",type=int,default=100000)
	parser.add_argument("--burn-in",dest="burn_in",type=int,default=20000)	
	parser.add_argument("--checkpoint-file",dest="checkpoint_file",type=str,default=None)
	parser.add_argument("--new-num-episodes",dest="new_num_episodes",type=int,default=None)
	parser.add_argument("--target-update-interval",dest="target_update_interval",type=int,default=None)
	
	return parser.parse_args()


def main(args):

	args = parse_arguments()

	environment_name = args.env
	num_episodes = args.num_episodes
	test_after = args.test_after
	lr = args.lr
	gamma = args.gamma
	replay_size = args.replay_size
	burn_in = args.burn_in
	save_data = 1
	save_after = 100
	checkpoint_file = args.checkpoint_file
	new_num_episodes = args.new_num_episodes
	num_trained_episodes = 0
	target_update_interval = args.target_update_interval

	ques_path = os.path.join(os.getcwd(),"duel_dqn") 
	env_path = os.path.join(ques_path,"env_{}".format(environment_name))
	curr_run_path = os.path.join(env_path,"num_ep_{}_lr_{}_gamma_{}".format(num_episodes,lr,gamma))
	plots_path = os.path.join(curr_run_path,"plots")
	data_path = os.path.join(curr_run_path,"data")

	if not os.path.isdir(ques_path):
		os.mkdir(ques_path)	

	if not os.path.isdir(env_path):
		os.mkdir(env_path)

	if not os.path.isdir(curr_run_path):
		os.mkdir(curr_run_path)

	if not os.path.isdir(plots_path):
		os.mkdir(plots_path)

	if not os.path.isdir(data_path):
		os.mkdir(data_path)

	env = gym.make(environment_name)
	
	## defining the Q_network
	Q_net = QNetwork(environment_name,env.observation_space,env.action_space,lr)
	DQ_net = QNetwork(environment_name,env.observation_space,env.action_space,lr)

	if(DEVICE.type=="cuda"):
		Q_net.cuda()
		print("model shifted to gpu")
	
	if(checkpoint_file):
		checkpoint_file = os.path.join(curr_run_path,checkpoint_file)
		num_trained_episodes = Q_net.load_checkpoint(checkpoint_file)
		num_episdoes = new_num_episodes

	agent = DQN_Agent(environment_name=environment_name, 
				Q_model = Q_net,
				DQ_model = DQ_net,
				render=False, 
				learning_rate=lr, 
				gamma=gamma, 
				replay_size=replay_size,
				burn_in=burn_in)
	
	train_loss = []
	mean_test_reward = [] 
	std_test_reward = []
	epsilon = 1.00
	update_target = False
	

	if (args.train):
		for ep in range(num_trained_episodes,num_episodes):

			update_target = (ep%target_update_interval)
			epsilon = max((1.00 - 0.95*ep/3000),0.05)
			
			train_loss.append(agent.train(epsilon,update_target))
			
			if(ep%test_after==0):
				print("episode : {}".format(ep))
				mean_reward, std_reward = agent.test(test_num_episodes=20)
				mean_test_reward.append(mean_reward)
				std_test_reward.append(std_reward)
				print("epsilon: {}".format(epsilon))
				print("mean: {}, std: {}".format(mean_reward,std_reward))
				
			if(save_data and ep%save_after==0):	
				np.save(os.path.join(data_path,"mean_test_reward.npy"),mean_test_reward)
				np.save(os.path.join(data_path,"std_test_reward.npy"),std_test_reward)
			if(ep%200==0):
				agent.Q_net.save_checkpoint(os.path.join(curr_run_path,"checkpoint.pth"),ep)
		
		fig = plt.figure(figsize=(16,9))
		plt.plot(train_loss,label="train_loss")
		plt.xlabel("num_stps")
		plt.ylabel("train_loss")
		plt.legend()
		plt.savefig(os.path.join(plots_path,"train_loss.png"))

		plt.plot(train_loss,label="train_loss")
		mean = np.array(mean_test_reward)
		std = np.array(std_test_reward)
		x = range(0,mean.shape[0])

		plt.clf()
		plt.close()
		fig = plt.figure(figsize=(16,9))
		plt.plot(mean,label="mean_cumulative_test_reward")
		plt.fill_between(x,mean-std,mean+std,facecolor="gray",alpha=0.5)
		plt.xlabel("num_training_episodes / 100")
		plt.ylabel("test_reward")
		plt.legend()
		plt.savefig(os.path.join(plots_path,"test_reward.png"))

if __name__ == '__main__':
	main(sys.argv)

