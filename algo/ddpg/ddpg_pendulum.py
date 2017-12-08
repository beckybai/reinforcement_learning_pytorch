import torch
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import datetime
from algo.ddpg import DDPGAgent
# import random
# from torch.autograd import Variable
import signal
import utils.ActionNoise

import os, sys, datetime, shutil
import utils.logger as logger
from datetime import  datetime
import utils.buffer as buffer
import gc


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
GAMMA = 0.99
steps_done = 0
animate = True
# MAX_STEP = 1000
noise = utils.ActionNoise.OrnsteinUhlenbeckActionNoise(1)


def run_episode(env, qf): # on line algorithm
	done = False
	global  steps_done
	obs = env.reset()
	add_reward = 0
	pending = []
	t = 0
	MAX_ITER = 300

	while( not done) and t<MAX_ITER:
		t += 1
		# if animate:
			# env.render()
		action = qf.select_action(obs.flatten())
		action_data = action.cpu().data.numpy().astype('float32') + 2*noise.sample() * env.action_space.high[0] # explotation
		# action_data = action.data.tolist()[0]
		obs_new, reward, done, _ = env.step(action_data)
		reward = reward
		qf.buffer.append([np.float32(obs),np.array(action_data),np.float32(obs_new),reward])
		add_reward+=reward
		if(done):
			obs_new =None
		# qf.trajectory.append({'reward':reward, 'state':obs, 'action':action,'new_state':obs_new})
		obs = obs_new
		qf.update() # picking data from a batch
		
		if done:
			break
	
	return add_reward

def run_policy(env, qf, episodes):
	total_steps = 0
	reward = []
	for e in range(1):
		reward.append(run_episode(env,qf))
		# qf.update() # update the policy net
		# qf.clear_trajectory() # clear the old trajectory

	return np.mean(reward)
	# print(np.mean(reward))
	# return reward

def main():
	torch.cuda.set_device(1)
	print(torch.cuda.current_device())
	seed_num = 1
	torch.cuda.manual_seed(seed_num)
	# buffer
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
	num_episodes = 1000
	rewards = np.zeros(num_episodes)
	identity = DDPGAgent.DDPGAgent(obs_dim, act_dim,max_clip,critic='TD', learning_rate=0.001,reward_decay = 0.99, e_greedy=0.9)
	for i_episode in range(num_episodes):
		rewards[i_episode] = run_policy(env,identity,episodes=100)
		print("In episode {}, the reward is {}".format(str(i_episode),str(rewards[i_episode])))
		gc.collect()
		if killer.kill_now:
			now = "dddpg_conti_TD_v0"
			identity.save_model(out_dir)
			break


	print('game over!')
	# util.before_exit(model=identity.model, reward=rewards)
	env.close()
	env.render(close=True)


if __name__ == "__main__":
	main()
