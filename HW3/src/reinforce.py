import sys
import argparse
import numpy as np
# import tensorflow as tf
# import keras
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import os
import math
# import seaborn as sns
# sns.set()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

class Policy(nn.Module):

	def __init__(self, nbActions, input_dim):
		super().__init__()

		## network
		#
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.linear1 = nn.Linear(input_dim,16)
		self.linear2 = nn.Linear(16,16)
		self.linear3 = nn.Linear(16,16)
		self.linear4 = nn.Linear(16,nbActions)



	def forward(self,X):

		X = X.to(device=DEVICE)
		x_em = F.relu(self.linear1(X))
		x_em = F.relu(self.linear2(x_em))
		x_em = F.relu(self.linear3(x_em))

		out = F.softmax(self.linear4(x_em),dim=1)

		return out




class Reinforce(object):
	# Implementation of the policy gradient method REINFORCE.

	def __init__(self, model, optimizer):

		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.model = model.to(device=DEVICE)
		self.model = model
		self.model.train()

		self.optimizer = optimizer

		# TODO: Define any training operations and optimizers here, initialize
		#       your variables, or alternately compile your model here.

	def action_to_one_hot(self,action,nbActions):
		action_vec = np.zeros((nbActions,))
		action_vec[action] = 1
		return action_vec


	def train(self, env, gamma=1.0):
		# Trains the model on a single episode using REINFORCE.
		# TODO: Implement this method. It may be helpful to call the class
		#       method generate_episode() to generate training data.
		
		states, actions, rewards = self.generate_episode(env)
		action_mask = torch.tensor(actions).bool().to(device=DEVICE)
		if(len(action_mask.shape)==1):
			action_mask = torch.unsqueeze(action_mask,dim=0)

		G = [None] * states.shape[0] ## return vector
		G[-1] = rewards[-1] * 0.01
		for i in range(states.shape[0]-2,-1,-1):
			G[i] = rewards[i] * 0.01 + gamma * G[i+1]

		G = torch.tensor(G).float().to(device=DEVICE)

		G_mean = torch.mean(G,dim=0).item()
		G_std = torch.std(G).item()

		G_normalised = (G - G_mean) / G_std

		action_probs = self.model(torch.tensor(states).float())
		action_probs = torch.masked_select(action_probs,action_mask)
		# selected_action_prob = torch.tensor([action_probs[0,a] for a in actions]).float().to(device=DEVICE)
		log_action_probs = torch.log(action_probs)

		loss = -1.0*torch.dot(G_normalised,log_action_probs) / states.shape[0]
		# loss = -1.0*torch.mean(G_normalised * log_action_probs) / states.shape[0]


		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		# print("train_loss: {}".format(loss.item()))

		return loss.item()

	def test(self,env,test_ep=20,argmax_tag=True):
		
		test_reward = []
		mean_test_reward = 0
		test_reward_var = 0
		self.model.eval()
		nbActions = env.action_space.n
		for ep in range(test_ep):
			curr_state = env.reset()
			done = False
			test_reward_per_episode = 0

			while not done:
				# with torch.no_grad():
				curr_state = torch.unsqueeze(torch.tensor(curr_state).float(),dim=0)
				
				if(argmax_tag):
					action = torch.argmax(self.model(curr_state),dim=1).item()
				else:
					action_probs = self.model(curr_state).detach().cpu().numpy()
					action = np.random.choice(action_probs.shape[1],size=1,p=action_probs[0])[0]

				next_state, reward, done, info = env.step(action)
				curr_state = next_state
				test_reward_per_episode += reward
		
			test_reward.append(test_reward_per_episode)

		mean_test_reward = np.mean(test_reward)
		test_reward_std = np.std(test_reward)
		
		print("mean_test_reward: {}".format(mean_test_reward))
		print("std_test_reward: {}".format(test_reward_std))

		self.model.train()
		return mean_test_reward, test_reward_std

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

		nbActions = env.action_space.n
		curr_state = env.reset()
		done = False

		while not done:
			states.append(curr_state)
		
			curr_state = torch.unsqueeze(torch.tensor(curr_state).float(),dim=0)
			with torch.no_grad():
				# action = torch.argmax(self.model(curr_state),dim=1).item()
				action_probs = self.model(curr_state).cpu().numpy()

			action = np.random.choice(action_probs.shape[1],size=1,p=action_probs[0])[0]
			# action = np.argmax(action_probs,axis=1)[0]
			
			actions.append(self.action_to_one_hot(action,nbActions))
			next_state, reward, done, info = env.step(action)
			curr_state = next_state
			rewards.append(reward)

		states = np.array(states)
		actions = np.array(actions) 
		rewards = np.array(rewards) 
		 

		return states, actions, rewards


def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000, help="Number of episodes to train on.")
	parser.add_argument('--lr', dest='lr', type=float,
						default=5e-4, help="The learning rate.")
	parser.add_argument('--gamma', dest='gamma', type=float,
						default=0.99, help="discount_factor")
	parser.add_argument('--env', dest='env', type=str,
						default='LunarLander-v2', help="environment_name")

	parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, 
						default=None, help="saved_checkpoint_file")

	parser.add_argument("--save-data-flag", dest="save_data_flag", type=int,
						default = 1, help="whether to save data or not")

	parser.add_argument("--save-checkpoint-flag", dest="save_checkpoint_flag", type=int,
						default = 1, help="whether to save checkpoint or not")

	parser.add_argument("--add-comment", dest="add_comment", type=str,
						default = None, help="any special comment for the model name")

	parser.add_argument("--use-argmax-test", dest="use_argmax_test", type=int,
						default = 0, help="using argmax during testing")



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
	lr = args.lr
	gamma = args.gamma
	render = args.render
	env_name = args.env
	save_checkpoint_flag = args.save_checkpoint_flag
	save_data_flag = args.save_data_flag
	checkpoint_file = args.checkpoint_file
	use_argmax_test = (True if args.use_argmax_test==1 else False)
	
	# create dir to store plots
	ques_path =	os.path.join(os.getcwd(),"reinforce") 
	env_path = os.path.join(ques_path,"env_{}".format(env_name))
	if(args.add_comment):
		curr_run_path = os.path.join(env_path,"num_ep_{}_lr_{}_gamma_{}_{}_test_argmax_{}".format(num_episodes,lr,gamma,args.add_comment,args.use_argmax_test))
	else:
		curr_run_path = os.path.join(env_path,"num_ep_{}_lr_{}_gamma_{}_test_argmax_{}".format(num_episodes,lr,gamma,args.use_argmax_test))
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


	# Create the environment.
	env = gym.make(env_name)
	# env = gym.make('LunarLander-v2')
	nbActions = env.action_space.n
	input_dim = env.observation_space.shape[0]
	test_after = 100

	# TODO: Create the model.
	policy = Policy(nbActions, input_dim)

	if(DEVICE.type=="cuda"):
		policy.cuda()
		print("model shifted to cuda")
	else:
		print("model shifted to cpu")


	for param in policy.parameters():
		if(len(param.shape)>1):
			alpha = np.sqrt(3.0*1.0 / ((param.shape[0] + param.shape[1])*0.5))
			nn.init.uniform_(param,-alpha, alpha)
		else:
			nn.init.constant_(param, 0.0)	


	optimizer = torch.optim.Adam(policy.parameters(),lr)
	
	num_episodes_trained = 0
	if(checkpoint_file):
		checkpoint = torch.load(os.path.join(curr_run_path,checkpoint_file))
		policy.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		num_episodes_trained = checkpoint["num_episodes_trained"]
		print("checkpoint_loaded: {}".format(checkpoint_file))
	
		
	train_loss_arr = []
	mean_test_reward_arr = []
	test_reward_std_arr = []
	
	best_test_reward = float('-inf')	
	
	# TODO: Train the model using REINFORCE and plot the learning curve.
	algo = Reinforce(policy,optimizer)

	for ep in range(num_episodes_trained, num_episodes):
		train_loss = algo.train(env,gamma)
		# print("episode: {}".format(ep))
		train_loss_arr.append(train_loss)
		if(ep%test_after==0):
			print("episode: {}".format(ep))
			mean_test_reward, test_reward_std = algo.test(env,test_ep=100,argmax_tag=use_argmax_test)
			mean_test_reward_arr.append(mean_test_reward)
			test_reward_std_arr.append(test_reward_std)
			if(best_test_reward<=mean_test_reward):
				best_test_reward = mean_test_reward * 1.0
			
		# saving_checkpoint
		if((ep+1)%1000==0 and save_checkpoint_flag):
			torch.save({'model_state_dict': policy.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'num_episodes_trained': ep},
					os.path.join(curr_run_path,"checkpoint.pth"))
			print("checkpoint_saved")
			
		if((ep+1)%1000==0 and save_data_flag):
			np.save(os.path.join(data_path,"train_loss.npy"),np.array(train_loss_arr))
			np.save(os.path.join(data_path,"mean_test_reward.npy"),np.array(mean_test_reward_arr))
			np.save(os.path.join(data_path,"std_test_reward.npy"),np.array(test_reward_std_arr))
			
			print("data_saved")

	fig = plt.figure(figsize=(16, 9))
	plt.plot(train_loss_arr)
	plt.xlabel("num episodes")
	plt.ylabel("train_loss")
	plt.savefig(os.path.join(plots_path,"train_loss_num_ep_{}_lr_{}_gamma_{}_use_argmax_{}.png".format(num_episodes,lr,gamma,args.use_argmax_test)))

	mean = np.array(mean_test_reward_arr)
	std = np.array(test_reward_std_arr)

	plt.clf()
	plt.close()
	fig = plt.figure(figsize=(16, 9))
	x =	np.arange(0,mean.shape[0])
	plt.plot(x,mean_test_reward_arr,label="mean_test_reward",color='coral')
	plt.fill_between(x,mean-std, mean+std,facecolor='peachpuff',alpha=0.5)
	# plt.errorbar(x=x ,
	# 			y=mean_test_reward_arr,
	# 			yerr=test_reward_std_arr,
	# 			ecolor='r',
	# 			capsize=10.0,
	# 			errorevery=5,
	# 			label='std')
	
	plt.xlabel("num episodes X {}".format(test_after))
	plt.ylabel("test_reward")
	plt.legend()
	plt.savefig(os.path.join(plots_path,"test_reward_num_ep_{}_lr_{}_gamma_{}_use_argmax_{}.png".format(num_episodes,lr,gamma,args.use_argmax_test)))

if __name__ == '__main__':
	main(sys.argv)
