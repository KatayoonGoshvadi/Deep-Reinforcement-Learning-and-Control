import sys
import argparse
import numpy as np
# import tensorflow as tf
# import keras
import gym
import torch
import torch.nn as nn 
import torch.nn.functional as F
import os
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.color import rgb2gray
from skimage.transform import resize
import pdb
# sns.set()

# from reinforce import Reinforce

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

class ActorNet(nn.Module):
	
	def __init__(self, InDim, NumA, actor_lr):
		
		super(ActorNet, self).__init__()
		

		self.conv1 = nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=True)
		self.conv2 = nn.Conv2d(1, 16, kernel_size=8, stride=4, bias=True)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, bias=True)
		
		self.fc1 = nn.Linear(9 * 9 * 32, 512, bias=True)
		self.fc2 = nn.Linear(512, NumA, bias=True)
		
		self.optimizer = torch.optim.Adam(self.parameters(),lr = actor_lr)
		
	def forward(self, X):
		X = X.to(device=DEVICE)
		x_em = F.relu(self.conv1(X))
		x_em = F.relu(self.conv2(x_em))
		x_em = F.relu(self.conv3(x_em))

		x_flat = x_em.view(X.shape[0],-1)

		x_flat = F.relu(self.fc1(x_flat))
		policy = F.softmax(self.fc2(x_flat),dim=1)
		
		return policy
	
class CriticNet(nn.Module):
	
	def __init__(self, InDim, critic_lr):
		
		super(CriticNet, self).__init__()
		
		self.conv1 = nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=True)
		self.conv2 = nn.Conv2d(1, 16, kernel_size=8, stride=4, bias=True)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, bias=True)
		
		self.fc1 = nn.Linear(9 * 9 * 32, 256, bias=True)
		self.fc2 = nn.Linear(256, 1, bias=True)
		
		self.optimizer = torch.optim.Adam(self.parameters(),lr = critic_lr)

		self.mse_loss = nn.MSELoss()

	def forward(self, X):
		X   = X.to(device=DEVICE)
		x_em = F.relu(self.conv1(X))
		x_em = F.relu(self.conv2(x_em))
		x_em = F.relu(self.conv3(x_em))

		x_flat = x_em.view(X.shape[0],-1)

		x_flat = F.relu(self.fc1(x_flat))

		value = self.fc2(x_flat)
		
		return value

class Replay_Memory():

	def __init__(self, memory_size=4):
		
		self.replay_deque = deque(maxlen=memory_size)
		self.memory_size = memory_size

	# def create_sample(self, batch_size=32):
	# 	# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
	# 	# You will feed this to your model to train.
	# 	sampled_ids = np.random.choice(np.arange(len(self.replay_deque)),size=batch_size)
		
	# 	sample = self.me
	# 	sampled_transitions = [self.replay_deque[id] for id in sampled_ids]

	# 	return sampled_transitions

	def append(self, transition):
		# Appends transition to the memory. 

		self.replay_deque.append(transition)
		pass


class A2C(object):
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	def __init__(self, actor_model, critic_model, n=20):
		# Initializes A2C.
		# Args:
		# - actor_model: The actor model.
		# - actor_lr: Learning rate for the actor model.
		# - critic_model: The critic model.
		# - critic_lr: Learning rate for the critic model.
		# - n: The value of N in N-step A2C.
		
		self.actor_model  = actor_model
		self.critic_model = critic_model
		self.n = n
		state_dq = deque(maxlen=4)

		# self.action_dq = deque(maxlen=4)

#         DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def action_to_one_hot(self,action,nbActions):
		action_vec = np.zeros((nbActions,))
		action_vec[action] = 1
		return action_vec
		
	def generate_episode(self, env, render=False):
		# Generates an episode by executing the current policy in the given env.
		# Returns:
		# - a list of states, indexed by time step
		# - a list of actions, indexed by time step
		# - a list of rewards, indexed by time step
		# TODO: Implement this method.
		states = []
		actions = []
		rewards = []
		state_dq = deque(maxlen=4)

		nbActions = env.action_space.n
		curr_screen = env.reset()
		done = False

		for i in range(3):
			state_dq.append(np.zeros(shape=(84,84)))

		curr_screen = self.preprocess(curr_screen)
		state_dq.append(curr_screen)
		
		curr_state = np.array(state_dq,dtype=np.float32)

		while not done:

			states.append(curr_state.copy()) ## appending states to the array

			curr_state = torch.from_numpy(np.expand_dims(curr_state,axis=0))

			with torch.no_grad():
				action_probs = self.actor_model.forward(curr_state).cpu().numpy()

			# action = np.random.choice(action_probs.shape[1],size=1,p=action_probs[0])[0]
			action = np.argmax(action_probs,axis=1)[0]

			actions.append(self.action_to_one_hot(action,nbActions))
			next_screen, reward, done, info = env.step(action)

			next_screen = self.preprocess(next_screen)
			state_dq.append(next_screen)

			next_state = np.array(state_dq,dtype=np.float32)

			curr_state = next_state.copy()

			rewards.append(reward)

		states = np.array(states)
		actions = np.array(actions) 
		rewards = np.array(rewards) 

		return states, actions, rewards

	def train(self, env, gamma=1.0):
		# Trains the model on a single episode using A2C.
		# TODO: Implement this method. It may be helpful to call the class
		#       method generate_episode() to generate training data.
		
		states, actions, rewards = self.generate_episode(env)
		
		rewards = rewards/100
			
		coeffs = [gamma**i for i in range(self.n)]
		

		
		Values = self.critic_model.forward(torch.from_numpy(states))

		R = np.empty(len(states))


  
		for i in range(1,len(states)+1):
			
			if i<=self.n:
				R[len(states)-i] = np.sum(coeffs[0:i]*rewards[len(states)-i:] )
				
			else:
				R[len(states)-i] = np.sum(coeffs * rewards[len(states)-i:len(states)-i+self.n]) + (gamma**self.n)*Values[len(states)-i+self.n].item()

		Vw = torch.squeeze(Values,dim=1)
		
		R = torch.tensor(R).float().to(device=DEVICE)
		action_mask = torch.tensor(actions).bool().to(device=DEVICE)

		if(len(action_mask.shape)==1):
			action_mask = torch.unsqueeze(action_mask,dim=0)
		

		action_probs = self.actor_model(torch.from_numpy(states))
		action_probs = torch.masked_select(action_probs,action_mask)
		
		log_action_probs = torch.log(action_probs)
		
		loss_actor = -1.0*torch.dot(R-Vw.clone().detach(),log_action_probs) / states.shape[0]
		loss_critic = self.critic_model.mse_loss(Vw,R)
		
		self.critic_model.optimizer.zero_grad()
		loss_critic.backward()
		self.critic_model.optimizer.step()

		self.actor_model.optimizer.zero_grad()
		loss_actor.backward()
		self.actor_model.optimizer.step()
		
		return loss_actor.item() , loss_critic.item()

	def test(self,env,test_ep=100):
		
		test_reward = []
		mean_test_reward = 0
		test_reward_var = 0

		state_dq = deque(maxlen=4)
		self.actor_model.eval()
		nbActions = env.action_space.n

		for ep in range(test_ep):

			# curr_state = env.reset()
			curr_screen = env.reset()
			done = False

			for i in range(3):
				state_dq.append(np.zeros(shape=(84,84)))
			

			curr_screen = self.preprocess(curr_screen)
			state_dq.append(curr_screen)

			curr_state = np.array(state_dq,dtype=np.float32)
			done = False
			test_reward_per_episode = 0

			while not done:
				curr_state = torch.from_numpy(np.expand_dims(curr_state,axis=0))
				# curr_state = self.preprocess(curr_state)
				# curr_state = torch.unsqueeze(torch.tensor(curr_state).float(),dim=0)
				with torch.no_grad():
					action_probs = self.actor_model.forward(curr_state).cpu().numpy()

				action = np.random.choice(action_probs.shape[1],size=1,p=action_probs[0])[0]

#                 curr_state = torch.unsqueeze(torch.tensor(curr_state).float(),dim=0)
#                 action = torch.argmax(self.actor_model(curr_state),dim=1).item()
		
				next_screen, reward, done, info = env.step(action)

				next_screen = self.preprocess(next_screen)
				state_dq.append(next_screen)
				next_state = np.array(state_dq,dtype=np.float32)

				curr_state = next_state
				
				test_reward_per_episode += reward
		
			test_reward.append(test_reward_per_episode)

		mean_test_reward = np.mean(test_reward)
		test_reward_std = np.std(test_reward)
		
		print("mean_test_reward: {}".format(mean_test_reward))
		print("std_test_reward: {}".format(test_reward_std))

		self.actor_model.train()
		return mean_test_reward, test_reward_std


	def preprocess(self,screen):
		screen = rgb2gray(screen)
		screen = resize(screen,(84,84),mode='reflect')
		# state = np.expand_dims(state,axis=0)

		return screen


def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=35000, help="Number of episodes to train on.")
	parser.add_argument('--actor-lr', dest='actor_lr', type=float,
						default=2e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=100, help="The value of N in N-step A2C.")
	parser.add_argument('--env', dest='env', type=str,
						default='Breakout-v0', help="environment_name")
	parser.add_argument('--gamma', dest='gamma', type=float,
						default=0.99, help="discount_factor")
	parser.add_argument("--save-data-flag", dest="save_data_flag", type=int,
						default = 1, help="whether to save data or not")


	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()


def main(args):
	# Parse command-line arguments.
	args = parse_arguments()
	num_episodes = args.num_episodes
	actor_lr = args.actor_lr
	critic_lr = args.critic_lr
	n = args.n
	render = args.render
	env_name = args.env
	gamma = args.gamma
	save_data_flag = args.save_data_flag


	# Create the environment.
	env = gym.make(env_name)
	nS = env.observation_space.shape[0]
	nA = env.action_space.n
	ques_path = os.path.join(os.getcwd(),"a2c_breakout")
	env_path = os.path.join(ques_path,"env_{}".format(env_name))
	curr_run_path = os.path.join(env_path,"num_ep_{}_lra_{}_lrc_{}_gamma_{}_n_{}".format(num_episodes,actor_lr,critic_lr,gamma,n))
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

	# TODO: Create the model.
	actor_model  = ActorNet(nS, nA, actor_lr)
	  
	critic_model = CriticNet(nS, critic_lr)

	if(DEVICE.type=="cuda"):
		actor_model.to(device=DEVICE)
		critic_model.to(device=DEVICE)
		print("models_shifted_to_gpu")

	#     if(torch.cuda.is_available()):
	#         actor_model.cuda()
	#         critic_model.cuda()

	
	for param in actor_model.parameters():
		if(len(param.shape)>1):
			alpha = np.sqrt(3.0*1.0 / ((param.shape[0] + param.shape[1])*0.5))
			nn.init.uniform_(param,-alpha, alpha)
		else:
			nn.init.constant_(param, 0.0)
			
	for param in critic_model.parameters():
		if(len(param.shape)>1):
			alpha = np.sqrt(3.0*1.0 / ((param.shape[0] + param.shape[1])*0.5))
			nn.init.uniform_(param,-alpha, alpha)
		else:
			nn.init.constant_(param, 0.0)

	algo = A2C(actor_model, critic_model, n=20)
	
	# TODO: Train the model using A2C and plot the learning curves.

	train_policy_loss_arr = []
	train_value_loss_arr = []

	mean_test_reward_arr = []
	test_reward_std_arr = []
	test_after = 100

	# TODO: Train the model using REINFORCE and plot the learning curve.

	for ep in range(0 ,num_episodes):
		train_policy_loss, train_value_loss = algo.train(env,gamma)        
		# print("episode: {}".format(ep))
		train_policy_loss_arr.append(train_policy_loss)
		train_value_loss_arr.append(train_value_loss)
		if(ep%test_after==0):
			print("episode: {}".format(ep))
			mean_test_reward, test_reward_std = algo.test(env)
			mean_test_reward_arr.append(mean_test_reward)
			test_reward_std_arr.append(test_reward_std)

		# saving_checkpoint
		# if((ep+1)%1000==0 and save_checkpoint_flag):
		#     torch.save({'model_state_dict': policy.state_dict(),
		#             'optimizer_state_dict': optimizer.state_dict(),
		#             'num_episodes_trained': ep},
		#             os.path.join(curr_run_path,"checkpoint.pth"))
		#     print("checkpoint_saved")

		if((ep+1)%1000==0 and save_data_flag):
			np.save(os.path.join(data_path,"train_policy_loss.npy"),np.array(train_policy_loss_arr))
			np.save(os.path.join(data_path,"train_value_loss.npy"),np.array(train_value_loss_arr))
			np.save(os.path.join(data_path,"mean_test_reward.npy"),np.array(mean_test_reward_arr))
			np.save(os.path.join(data_path,"std_test_reward.npy"),np.array(test_reward_std_arr))

			print("data_saved")

	fig1 = plt.figure(figsize=(16, 9))
	plt.plot(train_policy_loss_arr,label="training_policy_loss",color='red')
	plt.xlabel("num episodes")
	plt.ylabel("train_policy_loss")
	plt.legend()
	plt.savefig(os.path.join(plots_path,"train_policy_loss_num_ep_{}_lr_{}_gamma_{}_n_{}.png".format(num_episodes,actor_lr,gamma,n)))

	plt.clf()
	plt.close()

	fig2 = plt.figure(figsize=(16, 9))
	plt.plot(train_value_loss_arr,label="training_value_loss",color="indigo")
	plt.xlabel("num episodes")
	plt.ylabel("train_value_loss")
	plt.legend()
	plt.savefig(os.path.join(plots_path,"train_value_loss_num_ep_{}_lr_{}_gamma_{}_n_{}.png".format(num_episodes,critic_lr,gamma,n)))

	mean = np.array(mean_test_reward_arr)
	std = np.array(test_reward_std_arr)

	plt.clf()
	plt.close()
	
	fig3 = plt.figure(figsize=(16, 9))
	x = np.arange(0,mean.shape[0])
	plt.plot(x,mean, label="mean_test_reward",color='orangered')
	plt.fill_between(x,mean-std, mean+std,facecolor='peachpuff',alpha=0.5)

	plt.xlabel("num episodes X {}".format(test_after))
	plt.ylabel("test_reward")
	plt.legend()
	plt.savefig(os.path.join(plots_path,"test_reward_num_ep_{}_lr_a{}_lr_c{}_gamma_{}_n_{}.png".format(num_episodes,actor_lr,critic_lr,gamma, n)))

if __name__ == '__main__':
	main(sys.argv)
	
