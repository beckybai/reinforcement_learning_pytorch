import numpy as np
import torch
import model.QNet
from model.QNet import QNet
import random
from torch.autograd import Variable
import torch.nn.functional as F
from utils import util
import torch.optim as optim

class SarsaAgent(object):
	def __init__(self, state_dim, action_dim, learning_rate=0.001,reward_decay = 0.99, e_greedy=0.9):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = learning_rate
		self.gamma = reward_decay #  in according to the parameters in the formulation.
		self.epsilon = e_greedy
		self.EPS_START = 0.9
		self.EPS_END = 0.05
		self.EPS_DECAY = 100000 # this decay is to slow. # TO DO: figure out the relationship between the decay and the totoal step.
		# try to use a good strategy to solve this problem.
		use_cuda = torch.cuda.is_available()
		self.LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
		self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
		self.model = QNet(self.state_dim, self.action_dim).cuda() if use_cuda else QNet(self.state_dim, self.action_dim)

		self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
		# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5) # the learning rate decrease by a factor gamma every 10000 step_size.

		util.weights_init(self.model)

	def sbc(self,v,volatile=False):
		return Variable(self.FloatTensor((np.expand_dims(v,0).tolist())),volatile=volatile)

	def get_actions(self, state):
		action = self.model(self.sbc(state,volatile=True))
		return action


	def select_action(self, state,steps_done):
		util.adjust_learning_rate(self.optimizer, self.lr, steps_done, 10000)
		# global steps_done
		sample = random.random()
		esp_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
		                          np.exp(-1. * steps_done / self.EPS_DECAY)

		if sample > esp_threshold:
			actions = self.get_actions(state)
			action = actions.data.max(1)[1].view(1, 1)
			return action
		else:
			return self.LongTensor([[random.randrange(self.action_dim)]])

	def update(self, pending):#	def update(self, s, a, r, s_, a_,done=False):
		pending_len = len(pending)
		loss = 0
		while(pending_len):
			pending_len = pending_len -1
			[s,a,r,s_,a_,done] = pending[pending_len]
			if(done==True):
				expect_state_action_value = r
			else:
				non_final_next_states = self.model(self.sbc(s_,volatile=True))
				expect_state_action_value = r + self.gamma*non_final_next_states[0,a_]
				expect_state_action_value.volatile=False
			# expect_state_action_value = r + self.gamma*self.model(Variable(torch.from_numpy(np.expand_dims(s_,0).astype('float32')))).max(1)[0]
			state_action_value = self.model(self.sbc(s))[0,a]
			loss += 0.5*(state_action_value - expect_state_action_value).pow(2)
		self.optimizer.zero_grad()
		loss.backward()
		# loss.backward()
		# for param in self.model.parameters():
		# 	param.grad.data.clamp_(-1,1)
		self.optimizer.step()

	def save_model(self,name):
		self.model.save_state_dict(name)

	def load_model(self,name):
		self.model.load_state_dict(name)
