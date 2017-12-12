import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from model.QNet import ANet_policy
from model.VNet import VNet
import model.QNet as QNet
import model.VNet as VNet
import utils.util as util
import torch.nn.functional as F
import math
import utils.buffer as buffer

class ACAgent_con():
	def __init__(self, state_dim, action_dim, max_clip, critic_class='TD', learning_rate=0.001, reward_decay=0.99,
				 e_greedy=0.9):
		# load model or not
		# render or not
		torch.cuda.set_device(1)
		print(torch.cuda.current_device())
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = 0.0005
		# self.reward_dacy = reward_decay
		self.gamma = 0.95  # in according to the parameters in the formulation.
		self.epsilon = e_greedy
		self.EPS_START = 0.99
		self.EPS_END = 0.05
		self.EPS_DECAY = 100000  # this decay is to slow. # TO DO: figure out the relationship between the decay and the totoal step.
		# try to use a good strategy to solve this problem.
		self.use_cuda = torch.cuda.is_available()
		self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
		self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		self.max_clip = max_clip  # The output of the action should be in some range.
		self.critic_class = critic_class
		
		"""Model part"""
		# Policy Model $Pi$
		self.actor = ANet_policy(self.state_dim, self.action_dim,self.max_clip).cuda() if self.use_cuda else ANet_policy(
				self.state_dim, self.action_dim,self.max_clip)
		self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
		# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5) # the learning rate decrease by a factor gamma every 10000 step_size.
		self.actor.apply(util.weights_init)
		# util.weights_init(self.actor)  # The above is copy from SarsaAgent
		
		# State_Value Function V
		self.critic = VNet.VNet(self.state_dim, value_dim=1).cuda() if self.use_cuda else VNet.VNet(self.state_dim, value_dim=1)
		
		if(self.critic_class=='Q'):
			self.critic = VNet.Critic_net_Q(state_dim=state_dim, action_dim=action_dim).cuda() if self.use_cuda else VNet.Critic_net_Q(state_dim=state_dim,
																																	   action_dim=action_dim)
			
		self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

		# util.weights_init(self.modelV)
		self.critic.apply(util.weights_init)
		self.greedy = False
		self.trajectory = []
		# self.critic = 'Advantage' || 'TD(\labmda)Actor-Critic
		self.critic_class = critic_class
	
		MAX_BUFFER_SIZE = 1000000
		self.buffer = buffer.MemoryBuffer(size=MAX_BUFFER_SIZE)
		self.UD_BATCH_SIZE = 500  # update batch size
	
	# if(self.critic=='Q'):
	#     self.Critic = QNet_policy.
	
	
	def clear_trajectory(self):
		self.trajectory = []
	
	"""
	convert numpy format data to variable.
	"""
	
	def sbc(self, v, volatile=False):  # one-dimension
		if (len(np.shape(v)) == 1):
			return Variable(self.FloatTensor((np.expand_dims(v, 0).tolist())), volatile=volatile)
		elif (len(np.shape(v)) == 2):
			return Variable(self.FloatTensor(v), volatile=volatile)
		elif not np.shape(v):  # scalar
			return Variable(self.FloatTensor([v])).cuda() if self.use_cuda else Variable(torch.from_numpy([v]))
	
	def _sample_action(self, mu, std, batch_size):
		var = std.pow(2)
		eps = Variable(torch.randn(batch_size, self.action_dim)).cuda() if self.use_cuda else \
			Variable(torch.randn(batch_size, self.action_dim))
		return (mu + var * eps)
			# return torch.clamp(mu + var * eps, min=-self.max_clip, max=self.max_clip)
	
	def select_action(self, state):
		if (self.greedy):
			print("doesn't need to be greedy now")
			# TO DO
			return 0
		else:
			mu, std = self.actor(self.sbc(state))
			
			# mu, sigma = self.actor(self.sbc(state)).data.tolist()[0]
			assert std.data.tolist()[0][0] > 0
			# the batch size should be 1 in this formalation ?
			sampled_action = self._sample_action(mu=mu, std=std, batch_size=1)  # on-line....now
			return sampled_action.cuda() if self.use_cuda else sampled_action
	
	def discount_rewards(self, rewards):
		disrew = np.zeros(np.shape(rewards))
		rew = 0
		for i in reversed(range(len(rewards))):
			rew = rewards[i] + self.gamma * rew
			disrew[i] = rew
		return disrew
	def normal_log_density(self,x,mu,std):
		# Inputs are variables
		var = std.pow(2)
		log_density = -(x-mu).pow(2) / (2*var) - 0.5*np.log(2*math.pi) - torch.log(std)
		return log_density #? log_density.sum(1,keepdim=Ture)

	
	def update(self,t):

		if (self.critic_class == None):
			# discounted_reward = self.discount_rewards([point['reward'] for point in self.trajectory])
			#
			# critics = np.zeros([tra_size, 2])  # 2 means the dimension of actions
			# actions = []
			# for (i, point) in enumerate(self.trajectory):
			# 	critics[i, point['action']] = discounted_reward[i]
			# 	# states.append(point['state'])
			# 	states[i] = point['state']
			# 	actions.append(point['action'].tolist())
			# critics = self.sbc(critics)
			None
		
		elif (self.critic_class == 'REIN'):
			# batch consists of many trajectories. n * total_timestep(not fixed), batch_size is still one?
			# len_batch = len(self.trajectory)
			# batch = self.trajectory
			batch = self.buffer.sample(self.UD_BATCH_SIZE)
			len_batch = len(batch)
			# states = np.zeros([tra_size, self.state_dim])
			states = np.empty(shape=[len_batch, self.state_dim])
			actions = np.empty(shape=[len_batch, self.action_dim])
			next_states = np.empty(shape=[len_batch, self.state_dim])
			rewards = np.empty(shape=[len_batch, 1])
			
			for (i, sample) in enumerate(batch):
				states[i, :], actions[i, :], next_states[i, :], rewards[i] \
					= (sample[0].flatten()), (sample[1].flatten()), \
					  (sample[2].flatten()), (sample[3])
			dis_reward = self.discount_rewards(rewards)
			
			states = self.sbc(states, volatile=False)
			actions = self.sbc(actions)
			next_states = self.sbc(next_states)
			rewards = self.sbc(rewards)
			dis_reward = self.sbc(dis_reward)
			
			predicted = torch.squeeze(
					rewards + self.gamma * self.critic(next_states).detach()).detach()
			
			real = torch.squeeze(self.critic(states))
			critic_loss = F.smooth_l1_loss(real, predicted)
			self.optim_critic.zero_grad()
			critic_loss.backward()
			self.optim_critic.step()
			
			# update the actor
			# mu, sigma = self.actor(self.sbc(state))
			# # mu, sigma = self.actor(self.sbc(state)).data.tolist()[0]
			# assert sigma.data.tolist()[0][0] > 0
			# # the batch size should be 1 in this formalation ?
			# sampled_action = self._sample_action(mu=mu, var=sigma, batch_size=1)  # on-line....now
			# return sampled_action
			# action_value = self.select_action(states.cpu().data.tolist())
			mu, std = self.actor(states)
			# std = torch.exp(std)
			sampled_action = self._sample_action(mu=mu, std=std, batch_size=mu.data.shape[0])  # on-line....now
			prob = self.normal_log_density(sampled_action, mu, std)
			# actor_loss = torch.mean(-prob*(dis_reward.detach()-real.detach()))  # maximum the q function
			actor_loss = torch.mean(-prob * (real.detach()))  # maximum the q function
			# advantage function
			self.optim_actor.zero_grad()
			# print(actor_loss)
			actor_loss.backward()
			self.optim_actor.step()
		
		# for (i,point) in enumerate(self.trajectory):
		#     critics[i,point['action']] = self.QC(state) # for the convenience of calculation
		elif (self.critic_class == 'Q'):
			batch = self.buffer.sample(self.UD_BATCH_SIZE)
			len_batch = len(batch)
			# states = np.zeros([tra_size, self.state_dim])
			states = np.empty(shape=[len_batch, self.state_dim])
			actions = np.empty(shape=[len_batch, self.action_dim])
			next_states = np.empty(shape=[len_batch, self.state_dim])
			rewards = np.empty(shape=[len_batch, 1])
			
			for (i, sample) in enumerate(batch):
				states[i, :], actions[i, :], next_states[i, :], rewards[i] \
					= (sample[0].flatten()), (sample[1].flatten()), \
					  (sample[2].flatten()), (sample[3])
			dis_reward = self.discount_rewards(rewards)
			
			states = self.sbc(states, volatile=False)
			actions = self.sbc(actions)
			next_states = self.sbc(next_states)
			rewards = self.sbc(rewards)
			dis_reward = self.sbc(dis_reward)
			
			next_action = self.select_action(next_states.data.cpu().numpy().astype('float'))
			predicted = torch.squeeze(rewards + self.gamma * self.critic(next_states, next_action).detach()).detach()
			real = torch.squeeze(self.critic(states, actions))
			critic_loss = F.mse_loss(real, predicted)
			self.optim_critic.zero_grad()
			critic_loss.backward()
			self.optim_critic.step()
			
			mu,std = self.actor(states)
			# std = torch.exp(std)
			sampled_action = self._sample_action(mu=mu, std=std, batch_size=mu.data.shape[0])  # on-line....now
			prob = self.normal_log_density(sampled_action,mu,std)
			real = torch.squeeze(self.critic(states,sampled_action))
			# actor_loss = torch.mean(-prob*(dis_reward.detach()-real.detach()))  # maximum the q function
			actor_loss = torch.mean(-(real))  # maximum the q function
			# advantage function
			self.optim_actor.zero_grad()
			# print(actor_loss)
			actor_loss.backward()
			self.optim_actor.step()

		
		elif (self.critic_class == 'TD'):  # only use the value of the current state and the next state, also need a gamma,
			# batch consists of many trajectories. n * total_timestep(not fixed), batch_size is still one?
			# len_batch = len(self.trajectory)
			# batch = self.trajectory
			batch = self.buffer.sample(t)
			len_batch = len(batch)
			# states = np.zeros([tra_size, self.state_dim])
			states = np.empty(shape=[len_batch, self.state_dim])
			actions = np.empty(shape=[len_batch, self.action_dim])
			next_states = np.empty(shape=[len_batch, self.state_dim])
			rewards = np.empty(shape=[len_batch, 1])

			for (i, sample) in enumerate(batch):
				states[i, :], actions[i, :], next_states[i, :], rewards[i] \
					= (sample[0].flatten()), (sample[1].flatten()), \
					  (sample[2].flatten()), (sample[3])
			dis_reward = self.discount_rewards(rewards)

			states = self.sbc(states, volatile=False)
			actions = self.sbc(actions)
			next_states = self.sbc(next_states)
			rewards = self.sbc(rewards)
			dis_reward = self.sbc(dis_reward)
			
			predicted = torch.squeeze(
				rewards + self.gamma * self.critic(next_states).detach()).detach()
			
			real = torch.squeeze(self.critic(states))
			critic_loss = F.smooth_l1_loss(real, predicted)
			self.optim_critic.zero_grad()
			critic_loss.backward()
			self.optim_critic.step()
			
			# update the actor
			# mu, sigma = self.actor(self.sbc(state))
			# # mu, sigma = self.actor(self.sbc(state)).data.tolist()[0]
			# assert sigma.data.tolist()[0][0] > 0
			# # the batch size should be 1 in this formalation ?
			# sampled_action = self._sample_action(mu=mu, var=sigma, batch_size=1)  # on-line....now
			# return sampled_action
			# action_value = self.select_action(states.cpu().data.tolist())
			mu,std = self.actor(states)
			# std = torch.exp(std)
			sampled_action = self._sample_action(mu=mu, std=std, batch_size=mu.data.shape[0])  # on-line....now
			prob = self.normal_log_density(sampled_action,mu,std)
			real = torch.squeeze(self.critic(states))
			# actor_loss = torch.mean(-prob*(dis_reward.detach()-real.detach()))  # maximum the q function
			actor_loss = torch.mean(prob*(real.detach()))  # maximum the q function
			# advantage function
			self.optim_actor.zero_grad()
			# print(actor_loss)
			actor_loss.backward()
			self.optim_actor.step()
	
	# 	critic_tmp = self.sbc(r) + self.modelV(self.sbc(s_new, volatile=False)).detach() * self.gamma - self.modelV(
		# 		self.sbc(s, volatile=False))
		# 	td_error = torch.mean(critic_tmp.pow(2))
		# 	self.optimizerV.zero_grad()
		# 	td_error.backward()  # preserve the graph ?
		# 	self.optimizerV.step()
		# 	critics = critic_tmp  # detach?
		#
		# # policy_dis = self.actor(self.sbc(states))
		# loss = torch.sum(-torch.log((
		# 							action + 0.01)) * critics.detach())  # Action is a variable generated by actor model. We shouldn't generate it once more.
		# # What matter is if we can backpropagate this error. No idea...
		# self.optimizer.zero_grad()
		# loss.backward()
		# self.optimizer.step()

	def save_model(self,path):
		torch.save(self.actor.state_dict(), '{}/actor.pt'.format(path))
		torch.save(self.critic.state_dict(), '{}/critic.pt'.format(path))
		print('Models saved successfully')
