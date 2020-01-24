import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
import sys
import argparse

'''
python mod_plots.py --algo a2c_kati --env LunarLander-v2 --exp-dir num_ep_50000_lr_0.0005_gamma_0.99_test_argmax_0
'''
def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()


	parser.add_argument('--algo', dest='algo', type=str,
						default=None, help="algo name")

	parser.add_argument('--env', dest='env', type=str,
						default=None, help="name of the gym env")

	parser.add_argument('--exp-dir', dest='exp_dir',type=str,
							default=None, help='name of the ')

	return parser.parse_args()

def main(args):
	args = parse_arguments()

	algo = args.algo
	env = "env_{}".format(args.env)
	exp_dir = args.exp_dir

	data_dir = os.path.join(algo,env,exp_dir,"data")
	plot_dir = os.path.join(algo,env,exp_dir,"plots")

	mean = np.load(os.path.join(data_dir,"mean_test_reward.npy"))
	std = np.load(os.path.join(data_dir,"std_test_reward.npy"))

	x = range(0,mean.shape[0])

	figure = plt.figure(figsize=(16,9))

	plt.plot(x,mean,label="mean_cummulative_test_reward")
	plt.fill_between(x,mean-std, mean+std,facecolor='gray',alpha=0.5)

	plt.xlabel("num_training_episodes / 100")
	plt.ylabel("test_reward")
	plt.legend(loc="lower right")

	plt.savefig(os.path.join(plot_dir,"test_reward.png"))



if __name__== '__main__':
	main(sys.argv)