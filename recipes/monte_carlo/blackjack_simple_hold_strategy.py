import torch
import gym
from collections import defaultdict
from utils import plot_blackjack_value

env = gym.make("Blackjack-v0")

def run_episode(env, hold_score):
	state = env.reset()
	rewards = []
	states = [state]
	is_done = False
	while not is_done:
		action = 1 if state[0] < hold_score else 0
		state, reward, is_done, info = env.step(action)
		states.append(state)
		rewards.append(reward)
	return states, rewards

def mc_prediction_first_visit(env, hold_score, gamma, n_episode):
	V = defaultdict(float)
	N = defaultdict(int)
	for episode in range(n_episode):
		states_t, rewards_t = run_episode(env, hold_score)
		return_t = 0
		G = {}
		for state_t, reward_t in zip(states_t[1::-1], rewards_t[::-1]):
			return_t = gamma * return_t + reward_t
			G[state_t] = return_t
		for state, return_t in G.items():
			if state[0] <= 21:
				V[state] += return_t
				N[state] += 1
	for state in V:
		V[state] = V[state] / N[state]
	return V


if __name__ == "__main__":
	hold_score = 14
	gamma = 1
	n_episode = 500000

	V = mc_prediction_first_visit(env, hold_score, gamma, n_episode)
	print("Функция цености, вычисления методом МК первого посещения:\n", V)
	print("Количество состояний:", len(V))

	plot_blackjack_value(V)
