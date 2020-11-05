import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from utils import plot_blackjack_value

env = gym.make("Blackjack-v0")

def run_episode(env, Q, n_action):
	state = env.reset()
	rewards = []
	actions = []
	states = [state]
	is_done = False
	action = torch.randint(0, n_action, [1]).item()
	while not is_done:
		actions.append(action)
		states.append(state)
		state, reward, is_done, _ = env.step(action)
		rewards.append(reward)
		action = torch.argmax(Q[state]).item()
	return states, actions, rewards

def mc_control_on_policy(env, gamma, n_episode):
	"""
	Находит оптимальную стратегию методом управления МК с единой стратегией
	@return: оптимальная Q-функция и оптимальная стратегия
	"""
	n_action = env.action_space.n
	G_sum = defaultdict(float)
	N = defaultdict(int)
	Q = defaultdict(lambda: torch.empty(n_action))
	for episode in tqdm(range(n_episode), total=n_episode):
		states_t, actions_t, rewards_t = run_episode(env, Q, n_action)
		return_t = 0
		G = {}
		for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
			return_t = gamma * return_t + reward_t
			G[(state_t, action_t)] = return_t
		for state_action, return_t in G.items():
			state, action = state_action
			if state[0] <= 21:
				# Улучшение стратегии
				G_sum[state_action] += return_t
				N[state_action] += 1
				Q[state][action] = G_sum[state_action] / N[state_action]
	policy = {}
	# Оценивание стратегии
	for state, actions in Q.items():
		policy[state] = torch.argmax(actions).item()
	return Q, policy

if __name__ == "__main__":
	gamma = 1
	n_episode = 500000

	optimal_Q, optimal_policy = mc_control_on_policy(env, gamma, n_episode)
	print(optimal_Q)
	optimal_value = defaultdict(float)
	for state, action_values in optimal_Q.items():
		optimal_value[state] = torch.max(action_values).item()
	print(optimal_value)

	plot_blackjack_value(optimal_value)
