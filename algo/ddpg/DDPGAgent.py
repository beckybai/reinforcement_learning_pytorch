import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from model.QNet import ANet_policy
from model.VNet import VNet
import model.QNet as QNet
import model.VNet as VNet
import utils.util as util
import utils.buffer as buffer
import torch.nn.functional as F

class DDPGAgent():
	def __init__(self, state_dim, action_dim, action_lim, critic='TD', learning_rate=0.001, reward_decay=0.99,
				e_greedy=0.9):
		self.use_cuda = torch.cuda.is_available()


		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = 0.001
		self.gamma = reward_decay
		self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		
		# critic model
		self.critic = VNet.Critic_net_Q(state_dim=state_dim, action_dim=action_dim)
		# self.critic = QNet.Critic(state_dim=state_dim, action_dim=action_dim)
		self.target_critic = VNet.Critic_net_Q(state_dim=state_dim, action_dim=action_dim)
		# self.target_critic = QNet.Critic(state_dim=state_dim, action_dim=action_dim)
		
		self.optim_critic = torch.optim.Adam(self.critic.parameters(),self.lr)
		self.critic.apply(util.weights_init)
		util.weight_copy(self.critic,self.target_critic)
	
		# actor model
		self.actor = QNet.Actor_policy(state_dim=state_dim, action_dim=action_dim, action_lim= action_lim)
		# self.actor = QNet.Actor(state_dim=state_dim, action_dim=action_dim, action_lim= action_lim)
		self.target_actor = QNet.Actor_policy(state_dim=state_dim, action_dim=action_dim, action_lim= action_lim)
		# self.target_actor = QNet.Actor(state_dim=state_dim, action_dim=action_dim, action_lim= action_lim)
		self.optim_actor = torch.optim.Adam(self.actor.parameters(),self.lr)
		self.actor.apply(util.weights_init)
		util.weight_copy(self.actor, self.target_actor)
		
		# cuda
		self.critic = self.critic.cuda() if self.use_cuda else self.critic
		self.target_critic = self.target_critic.cuda() if self.use_cuda else self.target_critic
		self.actor = self.actor.cuda() if self.use_cuda else self.actor
		self.target_actor = self.target_actor.cuda() if self.use_cuda else self.target_actor
	
		# The buffer of the agent
		MAX_BUFFER_SIZE = 1000000
		self.buffer = buffer.MemoryBuffer(size = MAX_BUFFER_SIZE)
		self.UD_BATCH_SIZE = 100 # update batch size
	
	def sbc(self, v, volatile=False):  # one-dimension
		if (len(np.shape(v)) == 1):
			return Variable(self.FloatTensor((np.expand_dims(v, 0).tolist())), volatile=volatile)
		elif (len(np.shape(v)) == 2):
			return Variable(self.FloatTensor(v), volatile=volatile)
		elif not np.shape(v):  # scalar
			return Variable(self.FloatTensor([v])).cuda() if self.use_cuda else Variable(torch.from_numpy([v]))
	
	def select_action(self,state):
		# TODO add noise
		action = self.actor(self.sbc(state)).detach()
		return action # In C, the network generate the action is the targeted one which do not not to backpropogate the gradient
	
	def update(self):
		batch = self.buffer.sample(self.UD_BATCH_SIZE)
		len_batch = len(batch)
		states = np.empty(shape=[len_batch,self.state_dim])
		actions = np.empty(shape=[len_batch,self.action_dim])
		next_states = np.empty(shape=[len_batch,self.state_dim])
		rewards = np.empty(shape=[len_batch,1])
		for (i,sample) in enumerate(batch):
			states[i,:],actions[i,:],next_states[i,:],rewards[i]\
				= (sample[0].flatten()),(sample[1].flatten()), \
				  (sample[2].flatten()),(sample[3])
		
		states = self.sbc(states,volatile=False)
		actions = self.sbc(actions)
		next_states = self.sbc(next_states)
		rewards = self.sbc(rewards)
		# update the critic
		# difference = (rewards + self.target_critic( next_states,self.target_actor(next_states).detach()).detach()-
		# 					self.critic(states,actions))
		
		# critic_loss = torch.mean(torch.pow(difference,2))
		predicted = torch.squeeze(rewards + self.gamma*self.target_critic( next_states,self.target_actor(next_states).detach()).detach())
		real = torch.squeeze(self.critic(states,actions))
		critic_loss = F.smooth_l1_loss(real,predicted)
		self.optim_critic.zero_grad()
		critic_loss.backward()
		self.optim_critic.step()
		
		# update the actor
		action_value = self.critic(states.detach(),self.actor(states))
		actor_loss = -torch.mean(action_value) # maximum the q function
		self.optim_actor.zero_grad()
		# print(actor_loss)
		actor_loss.backward()
		self.optim_actor.step()
		
		util.weight_copy(self.actor, self.target_actor,tau=0.001) # mixed two weights...
		util.weight_copy(self.critic, self.target_critic, tau=0.001)
		
	
	def save_model(self,path):
		torch.save(self.target_actor.state_dict(), '{}/actor.pt'.format(path))
		torch.save(self.target_critic.state_dict(), '{}/critic.pt'.format(path))
		print('Models saved successfully')

