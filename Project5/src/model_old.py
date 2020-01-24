# import tensorflow as tf
# from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
# from keras.models import Model
# from keras.regularizers import l2
# import keras.backend as K
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import ZFilter
import pdb
import os 
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.num_nets   = num_nets
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), requires_grad = True).to(device=DEVICE)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), requires_grad = True).to(device=DEVICE)

        self.networks   = self.define_models()
        self.shift_networks_to_gpu(self.networks)
        # self.network = Network(state_dim, action_dim, learning_rate)

        # TODO write your code here
        # Create and initialize your mode

        self.plot_path = os.path.join(os.getcwd(),"plots")
        self.create_dirs(self.plot_path)
        self.save_path = os.path.join(os.getcwd(),"data_np")
        self.create_dirs(self.save_path)

    def shift_networks_to_gpu(self,networks):
        if(DEVICE.type=="cuda"):
            for net in networks:
                net.cuda()
        print("model shifted to gpu")

    def create_dirs(self,path):
        if not os.path.exists(path):
            os.mkdir(path)


    def define_models(self):

        models = []
        for i in range(self.num_nets):
            models.append(Network(self.state_dim,self.action_dim, self.learning_rate))

        return models

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    # def create_network(self):
    #     I = Input(shape=[self.state_dim + self.action_dim], name='input')
    #     h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
    #     h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
    #     h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
    #     O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
    #     model = Model(input=I, output=O)
    #     return model

    def lossFun(self, mean, cov, targets):

        # targets = torch.tensor(targets)
        return torch.sum((mean - targets)*(torch.reciprocal(cov))*(mean - targets), dim=1) + torch.log(torch.prod(cov,dim = 1))

    def predict_ns(self,inputs,model_num):

        # inputs = torch.tensor(inputs)
        # pdb.set_trace()

        outs = []

        with torch.no_grad():
            inputs = torch.tensor(inputs).to(device=DEVICE).float()
            for i,num in enumerate(model_num):
                out = self.networks[num](inputs[i])
                outs.append(out)

        outs = torch.stack(outs)
        # outs = torch.tensor(outs).to(device=DEVICE).float()

        mean , logvar =  self.get_output(outs)
        

        mean   = mean.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        std    = np.exp(logvar/2)

        ns = mean + std*np.random.normal(size = mean.shape)

        # ns = []
        # for i in range(mean.shape[0]):
        #     ns.append(np.random.multivariate_normal(mean[i], np.diag(var[i])) )

        return np.array(ns)

    def calc_rmse(self,mean,targets):
        # targets = torch.tensor(targets)
        err = torch.mean(torch.pow(mean-targets,2))
        err = torch.pow(err,0.5)
        return err

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        # TODO: write your code here\
        
        inputs = torch.tensor(inputs).to(device=DEVICE).float()
        targets = torch.tensor(targets).to(device=DEVICE).float()
        
        data_indices = np.arange(len(inputs))
        for n in range(self.num_nets):
            sampled_indices = np.random.choice(data_indices,len(data_indices),replace=True)

            epoch_loss_arr = []
            epoch_rmse_arr = []
            for e in range(epochs):
                np.random.shuffle(sampled_indices)
                num_data_points = len(sampled_indices)
                num_batches = max(1,len(sampled_indices) // batch_size)
                epoch_loss = 0
                rmse = 0
                for b in range(0,num_data_points,batch_size):
                    batch_indices = sampled_indices[b:b+batch_size]
                    input_batch   = inputs[batch_indices]
                    target_batch  = targets[batch_indices]

                    out = self.networks[n](input_batch)

                    mean , logvar = self.get_output(out)
                    
                    cov = torch.exp(logvar)

                    loss = self.lossFun(mean,cov,target_batch)

                    loss = torch.mean(loss)

                    self.networks[n].optimizer.zero_grad()
                    loss.backward()
                    self.networks[n].optimizer.step()
                    epoch_loss += loss.item()
                    rmse += self.calc_rmse(mean,target_batch).item()
                
                rmse /= num_batches
                epoch_loss /= num_batches
                epoch_rmse_arr.append(rmse*1.0)
                epoch_loss_arr.append(epoch_loss*1.0)
                print("Network: {} Epoch: {} epoch_loss: {} RMSE: {} ".format(n,e,epoch_loss,rmse))
                self.save_data(epoch_loss_arr,"loss",self.save_path)
                self.save_data(epoch_rmse_arr,"rmse",self.save_path)
            self.plot_prop(epoch_loss_arr,"loss",self.plot_path)
            self.plot_prop(epoch_rmse_arr,"rmse",self.plot_path)
            
        return epoch_loss_arr, epoch_rmse_arr

    def plot_prop(self,prop,prop_name,plots_path):
        fig = plt.figure(figsize=(16, 9))
        plt.plot(prop,label=prop_name,color='navy')
        plt.xlabel("epochs")
        plt.ylabel(prop_name)
        plt.legend()
        plt.savefig(os.path.join(plots_path,"{}.png".format(prop_name)))
        plt.clf()
        plt.close()

    def save_data(self,prop,prop_name,save_dir):
        np.save(os.path.join(save_dir,prop_name+".npy"),prop)

    # TODO: Write any helper functions that you need

class Network(nn.Module):


    def __init__(self,state_dim, action_dim, learning_rate):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l1   = nn.Linear((self.state_dim + self.action_dim), HIDDEN1_UNITS)
        self.l2   = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.l3   = nn.Linear(HIDDEN2_UNITS, HIDDEN3_UNITS)
        self.out  = nn.Linear(HIDDEN3_UNITS, 2 * self.state_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay = 0.0001)

    def forward(self,input):

        # input = torch.tensor(input).float()

        l1 = F.relu( self.l1(input) )
        l2 = F.relu( self.l2(l1) )
        l3 = F.relu( self.l3(l2) )
        out = self.out(l3)
        return out


