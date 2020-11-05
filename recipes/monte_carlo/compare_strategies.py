import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from blackjack_mc_qlearn import mc_control_on_policy

env = gym.make("Blackjack-v0")

hold_score = 18
hold_policy = {}
for player in range(2, 22):
	for dealer in range(1, 11):
		action = 1 if player < hold_score else 0
		hold_policy[(player, dealer, False)] = action
		hold_policy[(player, dealer, True)] = action

def simulate_episode(env , policy):
	state = env.reset()
	is_done = False
	while not is_done:
		action = policy[state]
		state, reward, is_done, info = env.step(action)
	return reward

n_episode = 100000
n_win_optimal = 0
n_win_simple = 0
n_lose_optimal = 0
n_lose_simple = 0

_, optimal_policy = mc_control_on_policy(env, gamma=1, n_episode=500000)

for _ in range(n_episode):
	reward = simulate_episode(env, optimal_policy)
	if reward == 1:
		n_win_optimal += 1
	elif reward == -1:
		n_lose_optimal += 1
	reward = simulate_episode(env, hold_policy)
	if reward == 1:
		n_win_simple += 1
	elif reward == -1:
		n_lose_simple += 1

print("Вероятность выигрыша при простой стратегии:", n_win_simple/n_episode)
print("Вероятность выигрыша при оптимальной стратегии:", n_win_optimal/n_episode)

print("Вероятность проигрыша при простой стратегии", n_lose_simple/n_episode)
print("Вероятность проигрыша при оптимальной стратегии", n_lose_optimal/n_episode)