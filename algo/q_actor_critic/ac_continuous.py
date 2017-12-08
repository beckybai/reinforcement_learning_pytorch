import torch
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import datetime
from ACAgent_con import ACAgent_con
# import random
# from torch.autograd import Variable
import signal
import utils.util as util

import os, sys, datetime, shutil
import utils.logger as logger
from datetime import  datetime


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display
plt.ion()


class GracefulKiller:
	def __init__(self):
		self.kill_now = False
		signal.signal(signal.SIGINT, self.exit_gracefully)
		signal.signal(signal.SIGTERM, self.exit_gracefully)

	def exit_gracefully(self,signum, frame):
		self.kill_now=True

# gym parameters
def init_gym(env_name):
	env = gym.make(env_name).unwrapped
	state_dim = env.observation_space.shape[0]
	disc_flag = len(env.action_space.shape)==0
	if disc_flag: # discrete action
		action_dim = env.action_space.n
	else:
		action_dim = env.action_space.shape[0]
	# ActionTensor = LongTensor if disc_flag else FloatTensor
	return env, state_dim, action_dim

BATCH_SIZE = 128
GAMMA = 0.999
steps_done = 0
animate = True
# MAX_STEP = 1000


def run_episode(env, qf): # on line algorithm
	done = False
	global  steps_done
	obs = env.reset()
	add_reward = 0
	pending = []
	t = 0
	MAX_ITER = 200

	while( not done) and t<MAX_ITER:
		t += 1
		# if animate:
			# env.render()
		action = qf.select_action(obs)
		action_data = action.data.tolist()[0]
		obs_new, reward, done, _ = env.step(action_data)
		reward = reward/10
		add_reward+=reward
		# qf.trajectory.append({'reward':reward, 'state':obs, 'action':action,'new_state':obs_new})
		obs = obs_new
		qf.update(obs,obs_new,action,reward) # On-Line

	return add_reward

def run_policy(env, qf, episodes):
	total_steps = 0
	reward = []
	for e in range(1):
		reward.append(run_episode(env,qf))
		# qf.update() # update the policy net
		qf.clear_trajectory() # clear the old trajectory

	return np.mean(reward)
	# print(np.mean(reward))
	# return reward

def main():
	torch.cuda.set_device(1)
	print(torch.cuda.current_device())
	seed_num = 1
	torch.cuda.manual_seed(seed_num)
#	data_dir = '/home/bike/data/mnist/'
	out_dir = '/home/becky/Git/reinforcement_learning_pytorch/log/AC_conti_{}/'.format(datetime.now())
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		shutil.copyfile(sys.argv[0], out_dir + '/TD_pendulum_v0.py')
	sys.stdout = logger.Logger(out_dir)
	env_name = 'Pendulum-v0'
	killer = GracefulKiller()
	env, obs_dim, act_dim = init_gym(env_name)
	max_clip = env.action_space.high[0]
	num_episodes = 300
	rewards = np.zeros(num_episodes)
	identity = ACAgent_con(obs_dim, act_dim, max_clip,critic='TD', learning_rate=0.0001,reward_decay = 0.99, e_greedy=0.9)
	for i_episode in range(num_episodes):
		rewards[i_episode] = run_policy(env,identity,episodes=100)
		print("In episode {}, the reward is {}".format(str(i_episode),str(rewards[i_episode])))
		if killer.kill_now:
			now = "AC_conti_TD_v0"
			identity.save_model(str(now))
			break


	print('game over!')
	util.before_exit(model=identity.model, reward=rewards)
	env.close()
	env.render(close=True)


if __name__ == "__main__":
	main()
