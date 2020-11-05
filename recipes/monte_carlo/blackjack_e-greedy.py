import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from utils import plot_blackjack_value, simulate_episode

env = gym.make("Blackjack-v0")

def run_episode(env, Q, epsilon, n_action):
	"""
	Выполняет эпизод следую e-жадной стратегии
	@return: результатирующие состояния, действия и вознаграждения для всего эпизода
	"""
	state = env.reset()
	rewards = []
	actions = []
	states = []
	is_done = False
	while not is_done:
		probs = torch.ones(n_action) * epsilon / n_action
		best_action = torch.argmax(Q[state]).item()
		probs[best_action] += 1. - epsilon
		action = torch.multinomial(probs, 1).item()
		actions.append(action)
		states.append(state)
		state, reward, is_done, _ = env.step(action)
		rewards.append(reward)
	return states, actions, rewards

def mc_control_epsilon_greedy(env, gamma, n_episode, epsilon):
	"""
	Строит оптимальную e-жадную стратегию методом управления МК с единой стратегией
	@param epsilon: компромисс между исследованием и использованием
	@return: оптимальные Q-функция и стратегия
	"""
	n_action = env.action_space.n
	G_sum = defaultdict(float)
	N = defaultdict(int)
	Q = defaultdict(lambda : torch.empty(n_action))
	for episode in tqdm(range(n_episode), total=n_episode):
		states_t, actions_t, rewards_t = run_episode(env, Q, epsilon, n_action)
		return_t = 0
		G = {}
		for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
			return_t = gamma * return_t + reward_t
			G[(state_t, action_t)] = return_t
		for state_action, return_t in G.items():
			state, action = state_action
			if state[0] <= 21:
				G_sum[state_action] += return_t
				N[state_action] += 1
				Q[state][action] = G_sum[state_action] / N[state_action]
	policy = {}
	for state, actions in Q.items():
		policy[state] = torch.argmax(actions).item()
	return Q, policy

gamma = 1
n_episode = 500000
epsilon = 0.1

optimal_Q, optimal_policy = mc_control_epsilon_greedy(env, gamma, n_episode, epsilon)
eval_policy(env, optimal_policy, optimal_Q)

