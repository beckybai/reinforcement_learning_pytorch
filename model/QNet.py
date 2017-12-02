import torch.nn as nn
import torch.nn.functional as F


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