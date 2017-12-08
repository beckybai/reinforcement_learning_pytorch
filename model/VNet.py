import torch.nn as nn
import torch.nn.functional as F

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
		self.h1 = 32
		self.h2 = 64
		self.fc1 = nn.Linear(state_dim,self.h1)
		self.fc2 = nn.Linear(self.h1, self.h2)
		self.fc3 = nn.Linear(self.h2, value_dim)


	def forward(self,x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		y = self.fc3(y)
		return y
