#!/usr/bin/env python
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np
import gym
import sys
import copy
import argparse
from collections import deque
import os
import random
import pdb
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


class QNetwork(tf.keras.Model):

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name,obs_space,action_space, lr, save_weights_path=None):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		#pdb.set_trace()

		super(QNetwork,self).__init__()
		decay_rate = lr / 10000000
		self.model = Sequential()
		self.save_weights_path = save_weights_path

	#	pdb.set_trace()
		input_dim = obs_space.shape[0]
		if (environment_name=="CartPole-v0"):
			self.model.add(Dense(30,input_dim=input_dim, activation='tanh')) #30 or  50
			# self.model.add(Dense(32,activation='relu'))		
			self.model.add(Dense(action_space.n))
		
		else:
			self.model.add(Dense(96,input_dim=input_dim, activation='relu'))
			self.model.add(Dense(96,activation='relu'))
			self.model.add(Dense(96,activation='tanh'))		
			self.model.add(Dense(action_space.n))
		
		adam = optimizers.Adam(lr=lr)
		self.model.compile(optimizer=adam,
				loss='mean_squared_error')
		

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		file_path = os.path.join(self.save_weights_path,suffix)		
		self.model.save_weights(file_path)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		
		self.model = tf.keras.load_model(os.path.join(self.save_weights_path,model_file))
		

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		self.model.load_weights(os.path.join(self.save_weights_path,weight_file))
		
		pass

	def custom_mse_loss(y_true,y_pred):
	## y_true: actual q_value of the state, chosen action
	## y_pred: q_values for all the functions for the corresponding state

		y_out = y_pred[0,actions]
		loss = keras.losses.mean_squared_loss(y_true,y_out) 
	
		return loss

		



class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		# Hint: you might find this useful:
		# 		collections.deque(maxlen=memory_size)
		
		# self.replay_buffer = []*memory_size
		
		self.replay_deque = deque(maxlen=memory_size)
		self.memory_size = memory_size
		self.burn_in = burn_in
		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		sampled_ids = np.random.choice(np.arange(len(self.replay_deque)),size=batch_size)
				
		sampled_transitions = [self.replay_deque[id] for id in sampled_ids]

		return sampled_transitions

	def append(self, transition):
		# Appends transition to the memory. 

		self.replay_deque.append(transition)
		pass


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False, learning_rate=1e-5, gamma=0.99, replay_size=100000, burn_in=20000, save_weights_path=None):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		
		self.environment_name = environment_name
		
		self.env = gym.make(self.environment_name)
		self.test_env = gym.make(self.environment_name) 

		self.obs_space = self.env.observation_space
		self.action_space = self.env.action_space
		
		self.Q_net = QNetwork(self.environment_name,self.obs_space, self.action_space, learning_rate, save_weights_path)
		self.Q_net_2 = QNetwork(self.environment_name,self.obs_space, self.action_space, learning_rate, save_weights_path)

		self.replay_buffer = Replay_Memory(memory_size=replay_size)
		self.burn_in_memory(burn_in)
		self.gamma = gamma
		self.batch_size = 32
		pass 

	def epsilon_greedy_policy(self, q_values, epsilon):
		# Creating epsilon greedy probabilities to sample from. 
	

		# go_greedy = np.random.choice(2,size=1,p=[epsilon,1-epsilon])[0]
		
		go_greedy = random.random()

		if(go_greedy > epsilon):
			action = np.argmax(q_values)
		else:
			action = np.random.choice(q_values.shape[1],size=1)[0]
		
		#action = np.argmax(q_values,axis=1) if go_greedy else np.random.choice(q_values.shape[1],size=1)
		return action

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		action = np.argmax(q_values)
		return action

	def train(self, num_episodes, test_after, eval_episodes,log_path=None):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		loss_per_step = []
		train_reward = []
		test_reward = []
		training_td_error = []

		update_target_after = 50 ## steps
		step_num = 0
		batch_loss = 0
		train_episode_reward = 0
		epsilon = 1.0

		self.Q_net.save_model_weights("initial_save.h5")
		self.Q_net_2.load_model_weights("initial_save.h5")

		for ep in range(num_episodes+1):
			# pdb.set_trace()

			curr_state = self.env.reset()
			done = False
			# epsilon = max((0.5 - 0.0495*ep / float(num_episodes)),0.05)
			# epsilon = max((1.0 - 2*0.95*ep / float(num_episodes)),0.05)
			# epsilon = max(0.999 * epsilon, 0.05)
			if(self.environment_name=="MountainCar-v0"):
				epsilon = max((1.0 - 2*0.95*ep / 5000),0.07)
			else:
				epsilon = max((1.0 - 2*0.95*ep / 3000),0.07)
			while not done:
				# epsilon = max((0.5 - 0.000000495*step_num),0.05)

				## select action using an epsilon greedy policy
				
				q_values = self.Q_net.model.predict(np.expand_dims(curr_state,axis=0))
				action = self.epsilon_greedy_policy(q_values,epsilon)
				
				## take a step in the env using the action
				next_state, reward, done, info = self.env.step(action)
				
				train_episode_reward += reward

				curr_state = next_state.copy()

				## store the transition in the replay buffer
				self.replay_buffer.append((curr_state,action,reward,next_state,done))
				
				## sample a minibatch of random transitions from the replay buffer
				sampled_transitions = self.replay_buffer.sample_batch(batch_size=self.batch_size)

				X_train = np.array([transition[0] for transition in sampled_transitions])
				transition_actions = np.array([transition[1] for transition in sampled_transitions])
				exp_rewards = np.array([transition[2] for transition in sampled_transitions])
				sampled_nxt_states = np.array([transition[3] for transition in sampled_transitions])
				dones = np.array([int(transition[4]) for transition in sampled_transitions])

				## updating the first Q network
				q_values_target = self.Q_net.model.predict(X_train)

				actions_nxt_state = np.argmax(self.Q_net.model.predict(sampled_nxt_states),axis=1)
				q_nxt_state = self.Q_net_2.model.predict(sampled_nxt_states)


				for sample_id in range(q_values_target.shape[0]):

					q_val_pred = q_values_target[sample_id, int(transition_actions[sample_id])]
					q_1_step = exp_rewards[sample_id] + self.gamma * q_nxt_state[sample_id,actions_nxt_state[sample_id]]
					training_td_error.append(q_val_pred - q_1_step)
					q_values_target[sample_id, int(transition_actions[sample_id])] = q_1_step

				history = self.Q_net.model.fit(X_train,q_values_target,verbose=0)
				step_num += 1

				if (step_num%update_target_after==0):
					self.Q_net.save_model_weights("temp.h5".format(ep,step_num))
					self.Q_net_2.load_model_weights("temp.h5".format(ep,step_num))

				loss_per_step.append(history.history['loss'][-1])
				
			# print("train_episode_reward: {}".format(train_episode_reward))
			
			train_reward.append(train_episode_reward)
			train_episode_reward = 0
			if((ep)%test_after==0):
				test_rew = self.test(test_num_episodes=eval_episodes)
				test_reward.append(test_rew)				

				print("Episode: {}".format(ep))
				print("epsilon: {}".format(epsilon))
				print("Test-----> Cum_reward: {}".format(test_rew))

				if(self.environment_name=="MountainCar-v0"):
					if(ep%400==0):
						self.Q_net.save_model_weights("episode{}_overall_step_{}.h5".format(ep,step_num))
						print("model_saved")


					if(ep>=1 and ep%400==0):
						np.save(os.path.join(log_path,"train_loss"), np.array(loss_per_step))
						np.save(os.path.join(log_path,"test_reward"), np.array(test_reward))
						np.save(os.path.join(log_path,"training_td_error"), np.array(training_td_error))
						np.save(os.path.join(log_path,"train_reward"), np.array(train_reward))
						
				else:
					if(ep==0 or ep==1600 or ep==3200 or ep==4500):
						self.Q_net.save_model_weights("episode{}_overall_step_{}.h5".format(ep,step_num))
						print("model_saved")	


		return loss_per_step, test_reward, training_td_error, train_reward

	def test(self, model_file=None, test_num_episodes=100):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		if(model_file):
			self.Q_net.load_model_weights(model_file)

		done = False
		cum_reward = 0
		epsilon = 0.05
		for episode in range(test_num_episodes):
			state_t = self.env.reset()
			done = False
			while not done:
				q_values = self.Q_net.model.predict(np.expand_dims(state_t,axis=0))
				action = self.epsilon_greedy_policy(q_values,epsilon)
				state_t_1, reward, done, info = self.env.step(action)
				cum_reward += reward
				state_t = state_t_1.copy()

		cum_reward /= test_num_episodes

		return cum_reward
		


	def burn_in_memory(self,burn_in):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		
		print("burn_in_start")

		transition_counter = 0
		nb_actions = self.action_space.n
		curr_state = self.env.reset()	
		done = False
		while transition_counter<burn_in:
		#	pdb.set_trace()		

			action = np.random.randint(0,nb_actions)
			
			next_state,reward,done,info = self.env.step(action)

			self.replay_buffer.append((curr_state,action,reward,next_state,done))
			transition_counter += 1
			if(done):
				curr_state = self.env.reset()
				done = False
			else:
				curr_state = next_state.copy()


		print("burn_in_over")
		pass


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi, weights_file, video_save_path):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
	#       	test_video(self, self.environment_name, episode)
	
	agent.Q_net.load_model_weights(weights_file)
	save_path = os.path.join(video_save_path,"videos-%s-%s" % (env, epi))
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	# To create video
	env = gym.wrappers.Monitor(agent.env, save_path, force=True)
	reward_total = []
	state = env.reset()
	done = False
	while not done:
		env.render()
		q_values = agent.Q_net.model.predict(np.expand_dims(state,axis=0))
		action = agent.epsilon_greedy_policy(q_values, 0.05)
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
	parser.add_argument('--model',dest='model_file',type=str,default=None)
	parser.add_argument("--lr",dest="lr",type=float,default=1e-5)
	parser.add_argument("--num-episodes",dest="num_episodes",type=int,default=1000)
	parser.add_argument("--test-after",dest="test_after",type=int,default=100)
	parser.add_argument("--eval-episodes",dest="eval_episodes",type=int,default=20)
	parser.add_argument("--gamma",dest="gamma",type=float,default=0.99)
	parser.add_argument("--replay-size",dest="replay_size",type=int,default=100000)
	parser.add_argument("--burn-in",dest="burn_in",type=int,default=20000)
	parser.add_argument("--vid-ep",dest="vid_ep",type=int,default=0)

	
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env
	num_episodes = args.num_episodes
	test_after = args.test_after
	eval_episodes = args.eval_episodes
	lr = args.lr
	gamma = args.gamma
	replay_size = args.replay_size
	burn_in = args.burn_in
	vid_ep = args.vid_ep

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)
	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	model_name = "{}_lr_{}_eps_{}_replay_sz_{}_burn_{}".format(args.env,args.lr,args.num_episodes,args.replay_size, args.burn_in)
	log_path = os.path.join(os.getcwd(),"keras_duel_dqn_"+model_name)
	save_weights_path = os.path.join(log_path,"saved_weights")
	video_save_path = os.path.join(log_path,"saved_videos")


	if not os.path.isdir(log_path):
	    os.mkdir(log_path)
	if not os.path.isdir(save_weights_path):
	    os.mkdir(save_weights_path)
	if not os.path.isdir(video_save_path):
	    os.mkdir(video_save_path)

	agent = DQN_Agent(environment_name=environment_name, 
						render=False, 
						learning_rate=lr, 
						gamma=gamma, 
						replay_size=replay_size,
						burn_in=burn_in,
						save_weights_path=save_weights_path)

	if (args.train):
		
		train_loss, test_reward, training_td_error, train_reward = agent.train(num_episodes, test_after, eval_episodes,log_path)

		np.save(os.path.join(log_path,"train_loss"), np.array(train_loss))
		np.save(os.path.join(log_path,"test_reward"), np.array(test_reward))
		np.save(os.path.join(log_path,"training_td_error"), np.array(training_td_error))
		np.save(os.path.join(log_path,"train_reward"), np.array(train_reward))
		
		plt.plot(train_loss)
		plt.xlabel("num_steps")
		plt.ylabel("train_loss")
		plt.savefig(os.path.join(log_path,"train_loss.png"))
		
		plt.clf()
		plt.close()
		plt.plot(test_reward)
		plt.xlabel("num_episodes")
		plt.ylabel("cummulative_test_reward")
		plt.savefig(os.path.join(log_path,"test_reward.png"))

		plt.clf()
		plt.close()
		plt.plot(training_td_error)
		plt.xlabel("num_steps")
		plt.ylabel("TD_error")
		plt.savefig(os.path.join(log_path,"training_td_error.png"))

		plt.clf()
		plt.close()
		plt.plot(train_reward)
		plt.xlabel("num_episodes")
		plt.ylabel("train_reward")
		plt.savefig(os.path.join(log_path,"train_reward.png"))

	else:
		weights_file = args.model_file
		test_video(agent, environment_name, vid_ep, weights_file, video_save_path)

if __name__ == '__main__':
	main(sys.argv)

