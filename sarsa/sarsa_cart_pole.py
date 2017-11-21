import torch
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from Q_function import QFunction
import random
from torch.autograd import Variable
import signal

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display
plt.ion()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

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


def run_episode(env, qf): # on line algorithm
	done = False
	global  steps_done
	obs = env.reset()
	# steps_done += 1
	action_new = qf.select_action(obs,steps_done)
	reward = 0

	while not done:
		if animate:
			env.render()
		action = action_new
		obs_new, rewards, done, _ = env.step(action[0,0])
		reward += rewards
		steps_done+=1
		action_new = qf.select_action(obs_new,steps_done)
		qf.update(obs,action[0,0],rewards, obs_new,action_new[0,0],done)
		obs = obs_new

	return reward

def run_policy(env, qf, episodes):
	total_steps = 0
	reward = []
	for e in range(episodes):
		reward.append(run_episode(env,qf))

	return np.mean(reward)
	# print(np.mean(reward))
	# return reward

def main():
	env_name = 'CartPole-v0'
	killer = GracefulKiller()
	env, obs_dim, act_dim = init_gym(env_name)
	num_episodes = 100
	rewards = np.zeros(num_episodes)
	QValue = QFunction(obs_dim, act_dim, learning_rate=0.01,reward_decay = 0.9, e_greedy=0.9)
	for i_episode in range(num_episodes):
		rewards[i_episode] = run_policy(env,QValue,episodes=100)
		print("In episode {}, the reward is {}".format(str(i_episode),str(rewards[i_episode])))
		if killer.kill_now:
			now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
			QValue.save_model(str(now))
			break

	print('game over!')
	env.close()
	env.render(close=True)


if __name__ == "__main__":
	main()

	# now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
