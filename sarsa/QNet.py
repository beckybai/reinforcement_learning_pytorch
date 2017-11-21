import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QNet,self).__init__()
		self.h1 = 128
		self.fc1 = nn.Linear(state_dim,self.h1)
		self.fc2 = nn.Linear(self.h1, action_dim)


	def forward(self,x):
		y = F.relu(self.fc1(x))
		y = F.sigmoid(self.fc2(y))
		return y

