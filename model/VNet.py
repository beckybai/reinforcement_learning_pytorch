import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

"""
ACAgent: modelV
ACAgent_con: Critic
Input:
	state
Output:
	value of that state
"""
class VNet(nn.Module):
	def __init__(self, state_dim, value_dim=1):
		super(VNet,self).__init__()
		self.h1 = 64
		self.h2 = 128
		self.fc1 = nn.Linear(state_dim,self.h1)
		self.fc2 = nn.Linear(self.h1, self.h2)
		self.fc3 = nn.Linear(self.h2, self.h2)
		self.fc4 = nn.Linear(self.h2, value_dim)
		self.initialize_weights()

	def initialize_weights(self):
		init.xavier_uniform(self.fc1.weight)
		init.xavier_uniform(self.fc2.weight)
		init.xavier_uniform(self.fc3.weight)
		init.xavier_uniform(self.fc4.weight)

	def forward(self,x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		y = F.relu(self.fc3(y))
		y = self.fc4(y)
		return y


"""
DDPGAgent:
Critic:
	Input: state, action
	Output: value
Actor:
	Input: state
	Ouput: probaility of that action
"""
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


"""
DDPGAgent
Input:
	state, action
Output:
	Q(s,a): scaler
"""


class Critic_net_Q(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic_net_Q, self).__init__()
		self.h1 = 64
		self.h2 = 128
		self.fc1_s = nn.Linear(state_dim, self.h1)
		self.fc2_s = nn.Linear(self.h1, self.h2)
		self.fc1_a = nn.Linear(action_dim, self.h1)
		# self.fc2_a = nn.Linear(self.h1, self.h2)
		
		self.fc1 = nn.Linear(self.h2 + self.h1, self.h1)
		self.fc2 = nn.Linear(self.h1, 1)
	
	def forward(self, state, action):
		s1 = F.elu(self.fc1_s(state))
		s1 = F.elu(self.fc2_s(s1))
		a1 = F.elu(self.fc1_a(action))
		# a1 = F.elu(self.fc2_a(a1))
		y1 = F.elu(self.fc1(torch.cat((s1, a1), dim=1)))  # concatinate the feature of action and the state.
		y1 = (self.fc2(y1))
		return y1
