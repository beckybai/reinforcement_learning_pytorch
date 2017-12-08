import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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
class ANet_policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(ANet_policy,self).__init__()
		self.h1 = 32
		self.h2 = 64
		self.fc1 = nn.Linear(state_dim, self.h1)
		self.fc2 = nn.Linear(self.h1, self.h2)
		self.mu = nn.Linear(self.h2, action_dim)
		self.sigma = nn.Linear(self.h2, action_dim)
	
	def forward(self, x):
		y = F.relu(self.fc1(x))
		y = F.sigmoid(self.fc2(y))
		mu = F.tanh(self.mu(y))
		sigma = F.softplus(self.sigma(y))
		return mu, sigma
	
"""
DDPGAgent
Input:
	state, action
Output:
	Q(s,a): scaler
"""
class Critic_net_Q(nn.Module):
	def __init__(self,state_dim, action_dim):
		super(Critic_net_Q,self).__init__()
		self.h1 = 128
		self.h2 = 256
		self.fc1_s = nn.Linear(state_dim, self.h1)
		self.fc2_s = nn.Linear(self.h1, self.h2)
		self.fc1_a = nn.Linear(action_dim, self.h1)
		# self.fc2_a = nn.Linear(self.h1, self.h2)
		
		self.fc1 = nn.Linear(self.h2 +self.h1, self.h1)
		self.fc2 = nn.Linear(self.h1, 1)
		
	def forward(self, state,action):
		s1 = F.elu(self.fc1_s(state))
		s1 = F.elu(self.fc2_s(s1))
		a1 = F.elu(self.fc1_a(action))
		# a1 = F.elu(self.fc2_a(a1))
		y1 = F.elu(self.fc1(torch.cat((s1,a1),dim=1))) # concatinate the feature of action and the state.
		y1 = (self.fc2(y1))
		return y1


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
		self.h1 = 256
		self.h2 = 512
		self.limit = action_lim
		self.fc1 = nn.Linear(state_dim, self.h1)
		self.fc2 = nn.Linear(self.h1, self.h2)
		self.fc3 = nn.Linear(self.h2, self.h1)
		self.fc4 = nn.Linear(self.h1, action_dim)
	
	def forward(self, x):
		y = F.elu(self.fc1(x))
		y = F.elu(self.fc2(y))
		y = F.elu(self.fc3(y))
		y = F.tanh(self.fc4(y))* self.limit
		return y
	
	

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,256)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(256,128)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim,128)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x


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



