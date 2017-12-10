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
import scipy.stats
import math
import copy
import collections.OrderedDict
from functools import reduce
from operator import mul

class TRPOAgent():
	def __init__(self, state_dim, action_dim, action_lim, critic='TD', learning_rate=0.001, reward_decay=0.99,
				 e_greedy=0.9):
		self.use_cuda = torch.cuda.is_available()
		# self.FloatTensor = torch.FloatTensor.cuda() if self.use_cuda else torch.FloatTensor
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = 0.001
		self.gamma = reward_decay
		self.action_lim = action_lim
		self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		
		# critic model
		self.critic = QNet.Policy_trpo(state_dim=state_dim, action_dim=action_dim)
		# self.optim_critic = torch.optim.Adam(self.critic.parameters(), self.lr)
		self.optim_critic = torch.optim.LBFGS(self.critic.parameters(),lr = learning_rate)
		self.critic.apply(util.weights_init)
		
		# actor model
		self.actor = QNet.Policy_trpo(state_dim=state_dim, action_dim=action_dim, action_lim=action_lim)
		self.optim_actor = torch.optim.Adam(self.actor.parameters(), self.lr)
		self.actor.apply(util.weights_init)
		
		# cuda
		self.critic = self.critic.cuda() if self.use_cuda else self.critic
		self.target_critic = self.target_critic.cuda() if self.use_cuda else self.target_critic
		self.actor = self.actor.cuda() if self.use_cuda else self.actor
		self.target_actor = self.target_actor.cuda() if self.use_cuda else self.target_actor
		
		# The buffer of the agent
		MAX_BUFFER_SIZE = 1000000
		self.buffer = buffer.MemoryBuffer(size=MAX_BUFFER_SIZE)
		self.UD_BATCH_SIZE = 100  # update batch size

		# Hessian-vector product parameters
		self.use_finite_differences = True # must be true if we want to use hessian-vector product to solve this problem
		self.cg_damping = 0.1
		self.cg_iters = 10
		self.residual_tol = 1e-8
		self.delta = 1e-2 # upper bound of kl divergence
		self.accept_radio = 1e-1 # bactracking line search
		self.max_backtracks = 10
		
		# parmaters reconstuction from a vector
		self.actor_properties = collections.OrderedDict()
		for k, v in self.actor.state_dict().items():
			self.actor_properties[k] = v.size()
			
		self.line_search = True # If false, then the algorithm is Natural Gradient Method
	
	
	def sbc(self, v, volatile=False):  # one-dimension
		if (len(np.shape(v)) == 1):
			return Variable(self.FloatTensor((np.expand_dims(v, 0).tolist())), volatile=volatile)
		elif (len(np.shape(v)) == 2):
			return Variable(self.FloatTensor(v), volatile=volatile)
		elif not np.shape(v):  # scalar
			return Variable(self.FloatTensor([v])).cuda() if self.use_cuda else Variable(torch.from_numpy([v]))
	
	def normal_log_density(self,x,mu,std):
		# Inputs are variables
		var = std.pow(2)
		log_density = -(x-mu).pow(2) / (2*var) - 0.5*torch.log(2*math.pi) - torch.log(std)
		return log_density #? log_density.sum(1,keepdim=Ture)
	
	
	def select_action(self, state):
		# TODO add noise
		mean_action, std_action = self.actor(self.sbc(state)).detach() # generate the mean and the variance
		actions = torch.normal(mean_action,std_action)
		# base_action = torch.randn(np.shape(state)[0]) # np.shape(state)[0] is the batch size
		# actions = torch.clamp(torch.pow(std_action,2)*base_action+mean_action, -self.action_lim,self.action_lim)
		action_probaility = scipy.stats.norm(mean_action.cpu().data.numpy(), std_action.cpu().data.numpy()).cdf(actions.cpu().data.numpy())
		return actions, action_probaility
		
		
	def discounted_rewards(self, rewards,dones): # The same as the one in the REINFORCEAgent
		disrew = np.zeros(np.shape(rewards))
		rew = 0
		for i in reversed(range(len(rewards))):
			if(dones[i]):
				rew = 0 # arrive at the end of the last trajectory
			rew = rewards[i] + self.gamma * rew
			disrew[i] = rew
		return disrew
	
	def flatten_params(self, parameters):
		return torch.cat([param.view(1,-1) for param in parameters],1) # form a one dimension vector
	
	def construct_model_from_theta(self, theta):
		"""
		Given a 1D parameter vector theta, return the policy model parameterized by theta
		"""
		theta = theta.squeeze(0)
		new_model = copy.deepcopy(self.actor)
		state_dict = collections.OrderedDict()
		start_index = 0
		for k, v in self.actor_properties.items():
			param_length = reduce(mul, v, 1)
			state_dict[k] = theta[start_index: start_index + param_length].view(v)
			start_index += param_length
		new_model.load_state_dict(state_dict)
		return new_model
	
	def calc_grad(self,model):
		return self.flatten_params([v.grad for v in model.parameters()]).squeeze(0) # form a one dimension vector
	
	def model_wrapper(self,model,states): # for simlicity
		mean,std = model(states)
		return torch.normal(mean,std)
	
	def kl_divergence(self, model,states):
		"""
		Returns an estimate of the average KL divergence between a given model and self.actor
		"""
		# observations_tensor = torch.cat(
		# 		[Variable(self.FloatTensor(state)).unsqueeze(0) for state in states])
		actprob = self.model_wrapper(model,states)
		# actprob = torch.normal(mean,std)
		# old_mean, old_std = model()
		old_actprob = self.model_wrapper(self.actor,states)
		
		# actprob = model(states)
		# old_actprob = self.actor(states)
		return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()
	
	def hessian_vector_product(self, vector,states):
		"""
		Returns the product of the Hessian of the KL divergence and the given vector
		"""
		# https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
		# Estimate hessian vector product using finite differences
		# Note that this might be possible to calculate analytically in future versions of PyTorch
		if self.use_finite_differences:
			r = 1e-6
			vector_norm = vector.data.norm()
			theta = self.flatten_params([param for param in self.actor.parameters()]).data
			
			model_plus = self.construct_model_from_theta(theta + r * (vector.data / vector_norm))
			model_minus = self.construct_model_from_theta(theta - r * (vector.data / vector_norm))
			
			kl_plus = self.kl_divergence(model_plus,states)
			kl_minus = self.kl_divergence(model_minus,states)
			kl_plus.backward()
			kl_minus.backward()
			
			grad_plus = self.calc_grad(model=model_plus)
			grad_minus = self.calc_grad(model = model_minus)
			# grad_plus = self.flatten_params([param.grad for param in model_plus.parameters()]).data
			# grad_minus = self.flatten_params([param.grad for param in model_minus.parameters()]).data
			damping_term = self.cg_damping * vector.data
			
			return vector_norm * ((grad_plus - grad_minus) / (2 * r)) + damping_term
		else:
			self.actor.zero_grad()
			kl_div = self.kl_divergence(self.actor,states)
			kl_div.backward(create_graph=True)
			gradient = self.flatten_params([v.grad for v in self.actor.parameters()])
			gradient_vector_product = torch.sum(gradient * vector)
			ones = torch.ones(gradient.size())
			if self.use_cuda:
				ones = ones.cuda()
			gradient_vector_product.backward(ones)
			return (self.flatten_params([v.grad for v in self.actor.parameters()]) - gradient).data
	
	def conjugate_gradient(self, b,states):
		"""
		Returns F^(-1)b where F is the Hessian of the KL divergence
		"""
		p = b.clone().data
		r = b.clone().data
		x = np.zeros_like(b.data.cpu().numpy())
		rdotr = r.dot(r)
		for i in range(self.cg_iters):
			z = self.hessian_vector_product(Variable(p),states).squeeze(0)
			v = rdotr / p.dot(z)
			x += v * p.cpu().numpy()
			r -= v * z
			newrdotr = r.dot(r)
			mu = newrdotr / rdotr
			p = r + mu * p
			rdotr = newrdotr
			if rdotr < self.residual_tol:
				break
		return x
	
	def surrogate_function(self, states, actions, model, advantage):
		# if(theta):
		# 	new_model = self.construct_model_from_theta(theta)
		mu, std = model(states)
		prob = self.normal_log_density(actions, mu, std)
		
		mu_old, std_old = self.actor(states)
		prob_old = self.normal_log_density(actions, mu_old, std_old)
		
		return -torch.sum(torch.exp(prob - prob_old) * advantage)
	
	def linesearch(self, theta_old, fullstep, expected_improve_rate, actions, states, advantage):
		model = self.construct_model_from_theta(theta_old)
		val_now = self.surrogate_function(states=states, actions=actions, model=model, advantage=advantage)
		for (i, step_alpha) in enumerate(.5 ** np.arange(self.max_backtracks)):
			theta_new = theta_old + step_alpha * fullstep
			model = self.construct_model_from_theta(theta_new)
			new_val = self.surrogate_function(states, actions, model, advantage)
			expected_improve = expected_improve_rate * step_alpha
			ratio = (new_val - val_now) / expected_improve_rate
			if ratio > self.accept_radio:
				return theta_new
		return theta_old

	
	def update(self,traj_len):
		batch = self.buffer.sample_continuous(traj_len)
		# batch = self.buffer.sample(self.UD_BATCH_SIZE)
		len_batch = len(batch)
		states = np.empty(shape=[len_batch, self.state_dim])
		actions = np.empty(shape=[len_batch, self.action_dim])
		next_states = np.empty(shape=[len_batch, self.state_dim])
		rewards = np.empty(shape=[len_batch, 1])
		dones = np.empty(shape=[len_batch,1])
		# action_distributions = np.empty(shape=[len_batch, self.state_dim])
		# action_distributions = torch
		for (i, sample) in enumerate(batch):
			states[i, :], actions[i, :], next_states[i, :], rewards[i], dones[i] \
				= (sample[0].flatten()), (sample[1].flatten()), \
				  (sample[2].flatten()), (sample[3]), \
				  (sample[4].flatten())
					
		states = self.sbc(states, volatile=False)
		actions = self.sbc(actions)
		next_states = self.sbc(next_states)
		discount_rewards = self.sbc(self.discounted_rewards(rewards,dones))
		"""
		Train the value network.
		"""
		def closure():
			self.optim_critic.zero_grad()
			output = self.critic(states)
			loss = F.mse_loss(output, discount_rewards)
			loss.backward()
			return loss
		# loss = F.mse_loss(value,discount_rewards)
		# self.optim_critic.zero_grad()
		# loss.backward()
		self.optim_critic.step(closure)
		
		"""
		Train the policy network.
		"""
		value = self.critic(states)
		advantage = discount_rewards - value # same as REINFORCEAgent, we use the discount_rewards to estimate Q(s,a)
		advantage_norm = (advantage - np.mean(advantage))/(np.std(advantage)+1e-8)
		
		mean_action, std_action = self.actor(states)
		state_probabilities = self.normal_log_density(actions,mean_action,std_action)
		fixed_probabilities = Variable(state_probabilities.data.clone()).detach()
		prob_ratio = torch.exp(state_probabilities - fixed_probabilities)
		surrogate_objective = torch.mean(prob_ratio* advantage_norm)
		"""
		Optimization Process
		maximize_theta g*(\theta - \theta_old)
		s.t.
			0.5*(\theta - \theta_old)^T * H * (\theta - \theta_old) <= \delta
		
		solution:
			s_unscale = \theta - \theta_old = 1/lambda * H ^(-1)*g  (1)
		
				(when doing the conjugate gradient descent, we need calculate the hessian-vector mutliplication.
			Fisher-Information-Matrix-vector multiplication is more efficient but harder to implement.)
			
			s = sqrt(2*max_kl/s_unscale * H * s_unscaled) * s_unscaled (2)
			
			back-tracking line search (3)
			
		"""
		
		self.optim_actor.zero_grad()
		surrogate_objective.backward()
		g = self.calc_grad(self.actor)
		step_direction = Variable(torch.from_numpy(self.conjugate_gradient(g,states)))
		
		# self.optim_actor.step()
		shs = (step_direction.dot(self.hessian_vector_product(step_direction,states))).cpu().data.numpy()
		step = step_direction*np.sqrt(2*self.delta/shs)
		
		obj_vale = g.dot(step)
		
		if(self.line_search):
			theta = self.linesearch( self.flatten_params([param.grad for param in self.actor.parameters()]),
									  step, obj_vale,actions,states,advantage)
		
		# why? diagnostics?
		old_model = copy.deepcopy(self.actor)
		old_model = old_model.load_state_dict(self.actor.state_dict())

		self.actor = self.construct_model_from_theta(theta)
		kl_old_new = self.kl_divergence(old_model) # for the convenience of debuging.
	
	def save_model(self, path):
		torch.save(self.target_actor.state_dict(), '{}/actor.pt'.format(path))
		torch.save(self.target_critic.state_dict(), '{}/critic.pt'.format(path))
		print('Models saved successfully')
