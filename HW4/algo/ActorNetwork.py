import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pdb

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300

def create_actor_network(state_size, action_size, batch_size, tau, learning_rate):
	"""Creates an actor network.

	Args:
		state_size: (int) size of the input.
		action_size: (int) size of the action.
	Returns:
		model: an instance of tf.keras.Model.
		state_input: a tf.placeholder for the batched state.
	"""
	actor_model = ActorNetwork(state_size,action_size, batch_size, tau, learning_rate)
	return actor_model

class ActorNetwork(nn.Module):
	def __init__(self, state_size, action_size, learning_rate,device):
		"""Initialize the ActorNetwork.
		This class internally stores both the actor and the target actor nets.
		It also handles training the actor and updating the target net.

		Args:
			sess: A Tensorflow session to use.
			state_size: (int) size of the input.
			action_size: (int) size of the action.
			batch_size: (int) the number of elements in each batch.
			tau: (float) the target net update rate.
			learning_rate: (float) learning rate for the critic.
		"""
		super(ActorNetwork, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.device = device

		self.actor = nn.Sequential(nn.Linear(state_size,HIDDEN1_UNITS),
									nn.ReLU(),
									nn.Linear(HIDDEN1_UNITS,HIDDEN2_UNITS),
									nn.ReLU(),
									nn.Linear(HIDDEN2_UNITS,action_size),
									nn.Tanh()
									)

		self.initialize_params()
		self.optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)

	def initialize_params(self):
                model_len = len(self.actor)
                for i,param in enumerate(self.parameters()):
                    if(i==model_len-2):
                        init_w = 3e-3
                    else:
                        init_w = np.sqrt(1.0 / param.shape[0])
                    nn.init.uniform_(param,-init_w,init_w)

	def forward(self,X):
		
		X = X.float().to(device=self.device)
		mu = self.actor(X) # mean action value of dimension (batch_size, action_size,)


		return mu
