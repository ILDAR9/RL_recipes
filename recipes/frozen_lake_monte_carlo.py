
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("FrozenLake-v0")
# env = gym.make("FrozenLake8x8-v0")
n_state = env.observation_space.n
print('states', n_state)
n_action = env.action_space.n
print('actions', n_action)

def run_episode(env, policy):
	state = env.reset()
	rewards = []
	states = [state]
	is_done = False
	while not is_done:
		action = policy[state].item()
		state, reward, is_done, info = env.step(action)
		states.append(state)
		rewards.append(reward)
	states = torch.tensor(states)
	rewards = torch.tensor(rewards)
	return states, rewards

def mc_prediction_first_visit(env, policy, gamma, n_episode):
	"""
	Оценивание стратегии методом Монте-Карло первого посещения
	"""
	V = torch.zeros(n_state)
	N = torch.zeros(n_state)
	for episode in range(n_episode):
		states_t, rewards_t = run_episode(env, policy)
		return_t = 0
		first_visit = torch.zeros(n_state)
		G = torch.zeros(n_state)
		for state_t, reward_t in zip(reversed(states_t)[1:], reversed(rewards_t)):
			return_t = gamma * return_t + reward_t
			G[state_t] = return_t
			first_visit[state_t] = 1
		for state in range(n_state):
			if first_visit[state] > 0:
				V[state] += G[state]
				N[state] += 1
	for state in range(n_state):
		if N[state] > 0:
			V[state] = V[state] / N[state]
	return V

def mc_prediction_every_visit(env, policy, gamma, n_episode):
	"""
	Оценивание стратегии методом Монте-Карло всех посещений
	"""
	V = torch.zeros(n_state)
	N = torch.zeros(n_state)
	G = torch.zeros(n_state)
	for episode in range(n_episode):
		states_t, rewards_t = run_episode(env, policy)
		return_t = 0
		for state_t, reward_t in zip(reversed(states_t[1:]), reversed(rewards_t)):
			return_t = gamma * return_t + reward_t
			G[state_t] += return_t
			N[state_t] += 1
	for state in range(n_state):
		if N[state] > 0:
			V[state] = G[state] / N[state]
	return V


gamma = 1
n_episode = 10000
# Заранее заготовленная оптимальная стратегия
optimal_policy = torch.tensor([0.,3.,3.,3.,0.,3.,2.,3.,3.,1.,0.,3.,3.,2.,1.,3.])

value = mc_prediction_first_visit(env, optimal_policy, gamma, n_episode)
print("Функция ценности, вычисленнаяметодом МК первого посещения:\n", value)
value = mc_prediction_every_visit(env, optimal_policy, gamma, n_episode)
print("Функция ценности, вычисленнаяметодом МК всех посещений:\n", value)

  