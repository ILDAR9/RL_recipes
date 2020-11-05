# Управление методом Монте-Карло с разделенной стратегией
import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from utils import plot_blackjack_value, simulate_episode, eval_policy

env = gym.make("Blackjack-v0")

def gen_random_policy(n_action):
	probs = torch.ones(n_action) / n_action
	def policy_function(state):
		return probs
	return policy_function

random_policy = gen_random_policy(env.action_space.n)

def run_episode(env, behavior_policy):
	"""
	Выполняет один эпизод, следуя заданной поведенческой стратегии
	@param behavior_policy: поведенческая стратегия
	@return:результатирующие состояния, действия и вознаграждения для всего эпизода
	"""
	state = env.reset()
	rewards = []
	states = []
	actions = []
	is_done = False
	while not is_done:
		probs = behavior_policy(state)
		action = torch.multinomial(probs, 1).item()
		actions.append(action)
		states.append(state)
		state, reward, is_done, _ =  env.step(action)
		rewards.append(reward)
	return states, actions, rewards

def mc_control_off_policy(env, gamma, n_episode, behavior_policy):
	"""
	Строит оптимальную стратегию методом управления МК с разделенной стратегией
	@param behavior_policy: поведенческая стратегия
	@return: оптимальные Q-функция и стратегия
	"""
	n_action = env.action_space.n
	G_sum = defaultdict(float)
	N = defaultdict(int)
	Q = defaultdict(lambda : torch.empty(n_action))
	for episode in tqdm(range(n_episode), total=n_episode):
		W = defaultdict(float)
		w = 1.
		states_t, actions_t, rewards_t = run_episode(env, behavior_policy)
		return_t = 0
		G = {}
		for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
			return_t = gamma * return_t + reward_t
			state_action = (state_t, action_t)
			G[state_action] = return_t
			W[state_action] = w
			if action_t != torch.argmax(Q[state_t]).item():
				break
			w *= 1./ behavior_policy(state_t)[action_t]
		for state_action, return_t in G.items():
			state, action = state_action
			if state[0] <= 21:
				G_sum[state_action] += return_t * W[state_action]
			N[state_action] += 1
			Q[state][action] = G_sum[state_action] / N[state_action]
	policy = {}
	for state, actions in Q.items():
		policy[state] = torch.argmax(actions).item()
	return Q, policy

def mc_control_off_policy_incremental(env, gamma, n_episode, behavior_policy):
	"""
	Строит оптимальную стратегию методом управления МК
	с разделенной стратегией инкрементно
	@param behavior_policy: поведенческая стратегия
	@return: оптимальные Q-функция и стратегия
	"""
	n_action = env.action_space.n
	N = defaultdict(int)
	Q = defaultdict(lambda: torch.empty(n_action))
	for episode in tqdm(range(n_episode), total = n_episode):
		W = 1.
		states_t, actions_t, rewards_t = run_episode(env, behavior_policy)
		return_t = 0.
		for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
			return_t = gamma * return_t + reward_t
			N[(state_t, action_t)] += 1
			Q[state_t][action_t] += (W / N[(state_t, action_t)]) * (return_t - Q[state_t][action_t])
			if action_t != torch.argmax(Q[state_t]).item():
				break
			W *= 1./ behavior_policy(state_t)[action_t]
	policy = {}
	for state, actions in Q.items():
		policy[state] = torch.argmax(actions).item()
	return Q, policy

def mc_control_off_policy_weighted(env, gamma, n_episode, behavior_policy):
	"""
	Строит оптимальную стратегию методом управления МК с разделенной
	стратегией и взвешенной  выборкой  по значимости
	@return: оптимальные Q-функция и стратегия
	"""
	n_action = env.action_space.n
	N = defaultdict(float)
	Q = defaultdict(lambda: torch.empty(n_action))
	for episode in tqdm(range(n_episode), total=n_episode):
		W = 1.
		states_t, actions_t, rewards_t = run_episode(env, behavior_policy)
		return_t = 0.
		for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
			return_t = gamma * return_t + reward_t
			N[(state_t, action_t)] += W
			Q[state_t][action_t] += (W / N[(state_t, action_t)]) * (return_t - Q[state_t][action_t])
			if action_t != torch.argmax(Q[state_t]).item():
				break
			W *= 1./ behavior_policy(state_t)[action_t]
	policy = {}
	for state, actions in Q.items():
		policy[state] = torch.argmax(actions).item()
	return Q, policy

gamma = 1
n_episode = 600000

# optimal_Q, optimal_policy = mc_control_off_policy(env, gamma, n_episode, random_policy)
# optimal_Q, optimal_policy = mc_control_off_policy_incremental(env, gamma, n_episode, random_policy)
optimal_Q, optimal_policy = mc_control_off_policy_weighted(env, gamma, n_episode, random_policy)
eval_policy(env, optimal_policy, optimal_Q)




