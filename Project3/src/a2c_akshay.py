import sys
import argparse
import numpy as np
#import tensorflow as tf
# import keras
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
from reinforce import Reinforce
import os
import pdb

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class a2cmodel(nn.Module):
	def __init__(self,input_dim,nbActions,lr):
		super().__init__()

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_net = nn.Sequential(nn.Linear(input_dim,16),
										nn.ReLU(),
										nn.Linear(16,16),
										nn.ReLU(),
										nn.Linear(16,16)
										)

		self.actor = nn.Sequential(nn.Linear(16, 8),
									nn.ReLU(),
									nn.Linear(8,nbActions),
									nn.Softmax(dim=1)
									)

		self.critic = nn.Sequential(nn.Linear(16, 8),
									nn.ReLU(),
									nn.Linear(8,1)
									)

		params_actor = list(self.state_net.parameters()) + list(self.actor.parameters())
		params_critic = list(self.state_net.parameters()) + list(self.critic.parameters())

		# self.optimizer_a = torch.optim.Adam(params_actor,lr_a)
		# self.optimizer_c = torch.optim.Adam(params_critic,lr_c)
		self.optimizer = torch.optim.Adam(self.parameters(),lr)




	def forward(self,X):
		X = X.to(device=DEVICE)
		
		state_em = self.state_net(X)

		policy = self.actor(state_em)

		value = self.critic(state_em)

		return policy,value

class A2C(Reinforce):
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	# def __init__(self, model, lr, critic_model, critic_lr, n=20):
	def __init__(self, model, n=20):
		# Initializes A2C.
		# Args:
		# - model: The actor model.
		# - lr: Learning rate for the actor model.
		# - critic_model: The critic model.
		# - critic_lr: Learning rate for the critic model.
		# - n: The value of N in N-step A2C.
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = model
		self.model.train()
		# self.critic_model = critic_model
		self.n = n
		self.mse_loss = nn.MSELoss()

		# TODO: Define any training operations and optimizers here, initialize
		#       your variables, or alternately compile your model here.
		
	


	def train(self, env, gamma=1.0):
		# Trains the model on a single episode using A2C.
		# TODO: Implement this method. It may be helpful to call the class
		#       method generate_episode() to generate training data.
		
		gamma_vec = np.array([gamma**i for i in range(0,self.n-1)])
		states, actions, rewards = self.generate_episode(env)
		T = states.shape[0] # episode horizon
		action_mask = torch.tensor(actions).bool().to(device=DEVICE)
		if(len(action_mask.shape)==1):
			action_mask = torch.unsqueeze(action_mask,dim=0)

		G = np.zeros((T,))
		action_probs,V_pred = self.model(torch.tensor(states,dtype=torch.float))

		# V_end = np.zeros((T,))
		# V_end[0:0:min(T,self.n-1)]
		# V_pred.clone().detach()


		for i in range(T-1,-1,-1):
			# print("i-----------",i)
			# print(gamma_vec[0:min(T,i+self.n-1)-i])
			# print(rewards[i+0:min(T,i+self.n-1)])
			temp = np.multiply(gamma_vec[0:min(T,i+self.n-1)-i],0.01*rewards[i+0:min(T,i+self.n-1)])
			G[i] = np.sum(temp)
			# G[i] = np.sum([(gamma**k)*rewards[k] if (i+k)<T else 0 for k in range(0,self.n-1)])
			if(i+self.n<T):
				G[i] += (gamma**self.n) * V_pred[i+self.n].item()



		G = torch.tensor(G,dtype=torch.float).to(device=DEVICE)

		G_mean = torch.mean(G,dim=0).item()
		G_std = torch.std(G).item()

		G_normalised = (G - G_mean) / G_std
		action_probs = torch.masked_select(action_probs,action_mask)
		log_action_probs = torch.log(action_probs)

		value_loss = self.mse_loss(torch.squeeze(V_pred,dim=1),G)
		advantage = G_normalised - torch.squeeze(V_pred.clone().detach(),dim=1)
		policy_loss = -1.0*torch.dot(advantage ,log_action_probs) / T

		# self.model.optimizer_a.zero_grad()
		# policy_loss.backward()
		# self.model.optimizer_a.step()
		
		# self.model.optimizer_c.zero_grad()
		# value_loss.backward()
		# self.model.optimizer_c.step()
		
		self.model.optimizer.zero_grad()
		(value_loss+policy_loss).backward()
		self.model.optimizer.step()
		

		return policy_loss.item(), value_loss.item()	

	def test(self,env,test_ep=20):
		
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
				action_probs,vals = self.model(curr_state)
				# action = torch.argmax(action_probs,dim=1).item()
				action_probs = action_probs.detach().cpu().numpy()
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
				action_probs,vals = self.model(curr_state)
				action_probs = action_probs.cpu().numpy()
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
						default=5e-4, help="The actor's learning rate.")
	# parser.add_argument('--lr-c', dest='lr_c', type=float,
	# 					default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=20, help="The value of N in N-step A2C.")
	parser.add_argument('--env', dest='env', type=str,
						default='LunarLander-v2', help="environment_name")
	parser.add_argument('--gamma', dest='gamma', type=float,
						default=0.99, help="discount_factor")
	parser.add_argument("--add-comment", dest="add_comment", type=str,
						default = None, help="any special comment for the model name")
	parser.add_argument("--save-data-flag", dest="save_data_flag", type=int,
						default = 1, help="whether to save data or not")
	parser.add_argument("--save-checkpoint-flag", dest="save_checkpoint_flag", type=int,
						default = 1, help="whether to save checkpoint or not")


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
	# lr_c = args.lr_c
	save_checkpoint_flag = args.save_checkpoint_flag
	save_data_flag = args.save_data_flag
	gamma = args.gamma
	env_name = args.env


	n = args.n
	render = args.render


	ques_path =	os.path.join(os.getcwd(),"a2c") 
	env_path = os.path.join(ques_path,"env_{}".format(env_name))
	if(args.add_comment):
		curr_run_path = os.path.join(env_path,"num_ep_{}_lr_{}_gamma_{}_n_{}_{}".format(num_episodes,lr,gamma,n,args.add_comment))
	else:
		curr_run_path = os.path.join(env_path,"num_ep_{}_lr_{}_gamma_{}_n_{}".format(num_episodes,lr,gamma,n))
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
	input_dim = env.observation_space.shape[0]
	nbActions = env.action_space.n


	# TODO: Create the model.
	model = a2cmodel(input_dim,nbActions,lr)	
	if(DEVICE.type=="cuda"):
		model.cuda()
		print("model shifted to cuda")
	else:
		print("model shifted to cpu")

	# pdb.set_trace()
	for param in model.parameters():
		if(len(param.shape)>1):
			alpha = np.sqrt(3.0*1.0 / ((param.shape[0] + param.shape[1])*0.5))
			nn.init.uniform_(param,-alpha, alpha)
		else:
			nn.init.constant_(param, 0.0)	
	algo = A2C(model,n)
	

	train_policy_loss_arr = []
	train_value_loss_arr = []

	mean_test_reward_arr = []
	test_reward_std_arr = []
	test_after = 100
	
	# TODO: Train the model using A2C and plot the learning curves.
	
	num_episodes_trained = 0
	for ep in range(num_episodes_trained ,num_episodes):
		train_policy_loss, train_value_loss = algo.train(env,gamma)
		# print("episode: {}".format(ep))
		train_policy_loss_arr.append(train_policy_loss)
		train_value_loss_arr.append(train_value_loss)

		if(ep%test_after==0):
			print("episode: {}".format(ep))
			mean_test_reward, test_reward_std = algo.test(env,test_ep=100)
			mean_test_reward_arr.append(mean_test_reward)
			test_reward_std_arr.append(test_reward_std)

			
		# saving_checkpoint
		# if((ep+1)%1000==0 and save_checkpoint_flag):
		# 	torch.save({'model_state_dict': policy.state_dict(),
		# 			'optimizer_state_dict': optimizer.state_dict(),
		# 			'num_episodes_trained': ep},
		# 			os.path.join(curr_run_path,"checkpoint.pth"))
		# 	print("checkpoint_saved")
			
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
	plt.savefig(os.path.join(plots_path,"train_policy_loss_num_ep_{}_lr_{}_gamma_{}.png".format(num_episodes,lr,gamma)))

	plt.clf()
	plt.close()

	fig2 = plt.figure(figsize=(16, 9))
	plt.plot(train_value_loss_arr,label="training_value_loss",color="indigo")
	plt.xlabel("num episodes")
	plt.ylabel("train_value_loss")
	plt.legend()
	plt.savefig(os.path.join(plots_path,"train_value_loss_num_ep_{}_lr_{}_gamma_{}.png".format(num_episodes,lr,gamma)))

	mean = np.array(mean_test_reward_arr)
	std = np.array(test_reward_std_arr)

	plt.clf()
	plt.close()

	fig3 = plt.figure(figsize=(16, 9))
	x =	np.arange(0,mean.shape[0])
	plt.plot(x,mean, label="mean_test_reward",color='orangered')
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
	plt.savefig(os.path.join(plots_path,"test_reward_num_ep_{}_lr_{}_gamma_{}.png".format(num_episodes,lr,gamma)))


if __name__ == '__main__':
	main(sys.argv)
