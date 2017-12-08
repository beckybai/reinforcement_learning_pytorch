import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from model.QNet import ANet_policy
from model.VNet import VNet
import utils.util as util


class ACAgent_con():
	def __init__(self, state_dim, action_dim, max_clip, critic='TD', learning_rate=0.001, reward_decay=0.99, e_greedy=0.9):
		# load model or not
		# render or not
		torch.cuda.set_device(1)
		print(torch.cuda.current_device())
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = learning_rate
		# self.reward_dacy = reward_decay
		self.gamma = reward_decay  # in according to the parameters in the formulation.
		self.epsilon = e_greedy
		self.EPS_START = 0.9
		self.EPS_END = 0.05
		self.EPS_DECAY = 100000  # this decay is to slow. # TO DO: figure out the relationship between the decay and the totoal step.
		# try to use a good strategy to solve this problem.
		self.use_cuda = torch.cuda.is_available()
		self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
		self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		self.max_clip = max_clip # The output of the action should be in some range.
		
		"""Model part"""
		# Policy Model $Pi$
		self.actor = ANet_policy(self.state_dim, self.action_dim).cuda() if self.use_cuda else ANet_policy(
			self.state_dim, self.action_dim)
		self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr*0.1)
		# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5) # the learning rate decrease by a factor gamma every 10000 step_size.
		util.weights_init(self.actor)  # The above is copy from SarsaAgent
		
		# State_Value Function V
		self.modelV = VNet(self.state_dim, value_dim=1).cuda() if self.use_cuda else VNet(self.state_dim, value_dim=1)
		self.optimizerV = optim.Adam(self.modelV.parameters(), lr=self.lr * 10)
		util.weights_init(self.modelV)
		self.greedy = False
		self.trajectory = []
		# self.critic = 'Advantage' || 'TD(\labmda)Actor-Critic
		self.critic = critic
	
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
		elif not np.shape(v): # scalar
			return Variable(self.FloatTensor([v])).cuda() if self.use_cuda else Variable(torch.from_numpy([v]))
			
	def _sample_action(self, mu, var, batch_size):
		eps = Variable(torch.randn(batch_size, self.action_dim)).cuda() if self.use_cuda else \
			Variable(torch.randn(batch_size, self.action_dim))
		return torch.clamp(mu + var*eps, min = -self.max_clip, max= self.max_clip)
	
	def select_action(self, state):
		if (self.greedy):
			print("doesn't need to be greedy now")
			# TO DO
			return 0
		else:
			mu, sigma = self.actor(self.sbc(state))
			# mu, sigma = self.actor(self.sbc(state)).data.tolist()[0]
			assert sigma.data.tolist()[0][0]>0
			# the batch size should be 1 in this formalation ?
			sampled_action = self._sample_action(mu=mu, var=sigma, batch_size=1) # on-line....now
			return sampled_action
			
	def discount_rewards(self, rewards):
		disrew = np.zeros(np.shape(rewards))
		rew = 0
		for i in reversed(range(len(rewards))):
			rew = rewards[i] + self.gamma * rew
			disrew[i] = rew
		return disrew
	
	def update(self, s, s_new, action, r):
		# batch consists of many trajectories. n * total_timestep(not fixed), batch_size is still one?
		tra_size = len(self.trajectory)
		states = np.zeros([tra_size, self.state_dim])
		if (self.critic == None):
			discounted_reward = self.discount_rewards([point['reward'] for point in self.trajectory])
			
			critics = np.zeros([tra_size, 2])  # 2 means the dimension of actions
			actions = []
			for (i, point) in enumerate(self.trajectory):
				critics[i, point['action']] = discounted_reward[i]
				# states.append(point['state'])
				states[i] = point['state']
				actions.append(point['action'].tolist())
			# critics = self.sbc(critics)
		
		elif (self.critic == 'Q'):
			# TO  DO
			None
		# for (i,point) in enumerate(self.trajectory):
		#     critics[i,point['action']] = self.QC(state) # for the convenience of calculation
		elif (self.critic == 'Adv'):
			None
		
		elif (self.critic == 'TD'):  # only use the value of the current state and the next state, also need a gamma,
			critic_tmp = self.sbc(r)+self.modelV(self.sbc(s_new,volatile=False)).detach()*self.gamma - self.modelV(self.sbc(s,volatile=False))
			td_error = torch.mean(critic_tmp.pow(2))
			self.optimizerV.zero_grad()
			td_error.backward() # preserve the graph ?
			self.optimizerV.step()
			critics = critic_tmp # detach?
			
		# policy_dis = self.actor(self.sbc(states))
		loss = torch.sum(-torch.log((action + 0.01)) * critics.detach()) # Action is a variable generated by actor model. We shouldn't generate it once more.
																# What matter is if we can backpropagate this error. No idea...
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
