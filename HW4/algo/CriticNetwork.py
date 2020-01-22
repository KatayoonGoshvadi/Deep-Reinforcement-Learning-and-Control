import numpy
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import pdb

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


class CriticNetwork(nn.Module):
        def __init__(self, state_size, action_size,learning_rate,device):
                """Initialize the CriticNetwork
                This class internally stores both the critic and the target critic
                nets. It also handles computation of the gradients and target updates.

                Args:
                        sess: A Tensorflow session to use.
                        state_size: (int) size of the input.
                        action_size: (int) size of the action.
                        batch_size: (int) the number of elements in each batch.
                        tau: (float) the target net update rate.
                        learning_rate: (float) learning rate for the critic.
                """
                super(CriticNetwork,self).__init__()
                self.state_size = state_size
                self.action_size = action_size
                self.device = device
                critic_input_dim = state_size + action_size
#                critic_input_dim = 2*((state_size + action_size)//2)
                # self.tau = tau

#               self.state_em = nn.Sequential(nn.Linear(state_size,(state_size+action_size)//2),
#                                                                       nn.ReLU(),
#                                                                       )
#
#               self.action_em = nn.Sequential(nn.Linear(action_size,(state_size+action_size)//2),
#                                                                       nn.ReLU(),
#                                                                       )

                self.critic = nn.Sequential(nn.Linear(critic_input_dim,HIDDEN1_UNITS),
                                                                        nn.ReLU(),
                                                                        nn.Linear(HIDDEN1_UNITS,HIDDEN2_UNITS),
                                                                        nn.ReLU(),
                                                                        nn.Linear(HIDDEN2_UNITS,1)
                                                                        )

                self.initialize_params()
                self.optimizer = torch.optim.Adam(self.parameters(),learning_rate)
                self.mse_loss = nn.MSELoss()
        
        def initialize_params(self):
            model_len = len(self.critic)
            for i,param in enumerate(self.parameters()):
                if(i==model_len-2):
                    init_w = 3e-4
                else:
                    init_w = np.sqrt(1.0/param.shape[0])
                nn.init.uniform_(param,-init_w,init_w)

        def forward(self,state,action):
                state = state.float().to(device=self.device)
                action = action.float().to(device=self.device)

#               s_em = self.state_em(state)
#               a_em = self.action_em(action) 
#                critic_input = torch.cat((s_em,a_em),dim=1))

                critic_input = torch.cat((state,action),dim=1)
                val = self.critic(critic_input)

                return val     

                
