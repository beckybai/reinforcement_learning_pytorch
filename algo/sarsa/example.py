import gym
# from tqdm import tqdm_notebook
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_random_policy():
	return np.random.choice(n_actions, tuple(bins))


def sample_reward(env, policy, t_max=200):
	s = env.reset()
	total_reward = 0

	for _ in range(t_max):
		discrete_s = tuple(discretize(s))
		s, reward, done, _ = env.step(policy[discrete_s])
		total_reward += reward
		if done:
			break

	return total_reward


def evaluate(policy, t_max=100, n_times=100):
	"""Run several evaluations and average the score the policy gets."""
	rewards = [sample_reward(env, policy, t_max) for _ in range(n_times)]
	return float(np.mean(rewards))


def crossover(policy1, policy2, p=0.5):
	"""

for each state, with probability p take action from policy1, else policy2
	"""
	randomness = np.random.choice([0, 1], policy1.shape, True, [p, 1 - p])
	return np.where(randomness == 0, policy1, policy2)


def mutation(policy, p=0.1):
	"""
	for each state, with probability p replace action with random action
	Tip: mutation can be written as crossover with random policy
	"""
	return crossover(policy, get_random_policy(), p)


def mutation_probability(policy_score, p_max=0.25, p_min=0.046875):
	return policy_score * (p_min - p_max) + p_max


env = gym.make('CartPole-v0')

n_actions = env.action_space.n
t_max = 200
n_times = 10

bins = [11, 11, 11, 11]
modules = env.observation_space.high
modules[1] = 5
modules[3] = 5


def discretize(continuous_state):
	discrete_state = np.zeros_like(continuous_state)
	for i in range(continuous_state.size):
		gist = np.linspace(-modules[i], modules[i], bins[i])
		discrete_state[i] = np.abs(gist - continuous_state[i]).argmin()
	return discrete_state.astype(int)


n_epochs = 10
pool_size = 300
n_crossovers = 150
n_mutations = 150
pool = [get_random_policy() for _ in range(pool_size)]
pool_scores = list(map(evaluate, pool, np.full(len(pool), t_max, np.int), np.full(len(pool), n_times, np.int)))

scores = []
for epoch in range(n_epochs):
	print("Epoch %s:" % epoch)

	norm = sum(pool_scores) if sum(pool_scores) > 0 else 1
	rands1 = np.random.choice(pool_size, n_crossovers, [e / norm for e in pool_scores])
	rands2 = np.random.choice(pool_size, n_crossovers, [e / norm for e in pool_scores])
	probs = [pool_scores[i] / (pool_scores[i] + pool_scores[j]) if pool_scores[i] + pool_scores[j] > 0 else 0.5
	         for i, j in zip(rands1, rands2)]
	crossovered = [crossover(pool[i], pool[j], p)
	               for i, j, p in zip(rands1, rands2, probs)]
	rands = np.random.randint(0, pool_size, n_mutations)
	mutated = [mutation(pool[i], mutation_probability(pool_scores[i] / norm)) for i in rands]

	pool = pool + crossovered + mutated
	pool_scores = list(map(evaluate, pool, np.full(len(pool), t_max, np.int), np.full(len(pool), n_times, np.int)))

	selected_indices = np.argsort(pool_scores)[-pool_size:]
	pool = [pool[i] for i in selected_indices]
	pool_scores = [pool_scores[i] for i in selected_indices]

	print(''.join(['best score:', str(pool_scores[-1]), ', mean score: ', str(np.mean(pool_scores))]))
	scores.append(pool_scores[-1])

env.close()
