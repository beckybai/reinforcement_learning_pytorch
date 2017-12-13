import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init

EPS = 0.003


def fanin_init(size):
	fanin =  size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).normal_(0,v)


"""
QTDAgent
Input:
	state
Output:
	action
"""
class QNet(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QNet,self).__init__()
		self.h1 = 32
		self.h2 = 64
		self.fc1 = nn.Linear(state_dim,self.h1)

		self.fc2 = nn.Linear(self.h1, self.h2)

		self.fc3 = nn.Linear(self.h2, action_dim)


	def forward(self,x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		y = self.fc3(y)
		return y

"""
REINFORCEAgent
Input:
	state
Output:
	action
	(things are different for discrete one and continuous one)
	For continuous one, the output is directly an action.
	
	softmax
"""
class QNet_policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QNet_policy,self).__init__()
		self.h1 = 32
		self.h2 = 64
		self.fc1 = nn.Linear(state_dim,self.h1)
		self.fc2 = nn.Linear(self.h1, self.h2)
		self.fc3 = nn.Linear(self.h2, action_dim)


	def forward(self,x):
		y = F.relu(self.fc1(x))
		y = F.sigmoid(self.fc2(y))
		y = F.softmax(self.fc3(y))
		return y
	
"""
ACAgent_continous
Input:
	state
Output:
	distribution of a contuous action with normal distribution (\mu, \sigma)
	Using the reparametric trick later.
"""
EPS = 0.001
class ANet_policy(nn.Module):
	def __init__(self, state_dim, action_dim, action_lim):
		super(ANet_policy,self).__init__()
		self.action_lim = action_lim
		self.h1 = 64
		self.h2 = 128
		self.fc1 = nn.Linear(state_dim, self.h1)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(self.h1, self.h2)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		
		self.fc3 = nn.Linear(self.h2, self.h2)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.mu = nn.Linear(self.h2, action_dim)
		self.mu.weight.data.uniform_(-1/(2*self.h2),1/(2*self.h2))


		self.sigma = nn.Linear(self.h2, action_dim)
		self.sigma.weight.data.uniform_(-1/(2*self.h2),1/(2*self.h2))

	
	def forward(self, x):
		y = F.sigmoid(self.fc1(x))
		y = F.tanh(self.fc2(y))
		y = F.tanh(self.fc3(y))
		mu = F.tanh(self.mu(y))*2
		sigma = torch.exp(self.sigma(y))-torch.exp(self.sigma(y))+0.2
		return mu, sigma
	


"""
DDPGAgent
Input:
	state
Output:
	action
	(things are different for discrete one and continuous one)
	For continuous one, the output is directly an action.

	tanh ( the output range is [-2,2] )
"""

class Actor_policy(nn.Module):
	def __init__(self, state_dim, action_dim,action_lim):
		super(Actor_policy, self).__init__()
		self.h1 = 64
		self.h2 = 128
		self.limit = action_lim
		self.fc1 = nn.Linear(state_dim, self.h1)
		self.fc2 = nn.Linear(self.h1, self.h2)
		self.fc3 = nn.Linear(self.h2, self.h1)
		self.fc4 = nn.Linear(self.h1, action_dim)
	
	def forward(self, x):
		y = F.elu(self.fc1(x))
		y = F.elu(self.fc2(y))
		y = F.elu(self.fc3(y))
		y = F.tanh(self.fc4(y))* 2
		return y
	


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,256)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = F.tanh(self.fc4(x))

		action = action * self.action_lim

		return action


"""
"""

class Policy_trpo(nn.Module):
	
	# def __init__(self):
	
	def __init__(self,state_dim, action_dim,action_lim):
		# super(Policy_trpo)
		super(Policy_trpo,self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,256)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.mean = nn.Linear(64,action_dim)
		self.mean.weight.data.uniform_(-EPS,EPS)
		
		self.variance = nn.Linear(64, action_dim)
		self.variance.weight.data.uniform_(-EPS,EPS)
		self.initialize()
	
	def initialize(self):
		init.xavier_uniform(self.fc1.weight)
		init.xavier_uniform(self.fc2.weight)
		init.xavier_uniform(self.fc3.weight)
		init.xavier_uniform(self.mean.weight)
		init.xavier_uniform(self.variance.weight)
		
	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action_mean = F.tanh(self.mean(x))
		action_std = F.softmax(self.variance(x))
		return action_mean, action_std