import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from collections import defaultdict

def plot_surface(X, Y, Z, title):
	fig = plt.figure(figsize=(20, 10))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
						   cmap=matplotlib.cm.coolwarm, vmin=-1., vmax=1.)
	ax.set_xlabel("Очки игрока")
	ax.set_ylabel("Открытая карта сдающего")
	ax.set_zlabel("Ценность")
	ax.set_title(title)
	ax.view_init(ax.elev, -120)
	fig.colorbar(surf)
	plt.show()

def plot_blackjack_value(V):
	player_sum_range = range(12, 22)
	dealer_show_range = range(1, 11)
	X, Y = torch.meshgrid([torch.tensor(player_sum_range), torch.tensor(dealer_show_range)])
	values_to_plot = torch.zeros((len(player_sum_range), len(dealer_show_range), 2))
	for i, player in enumerate(player_sum_range):
		for j, dealer in enumerate(dealer_show_range):
			for k, ace in enumerate([False, True]):
				values_to_plot[i, j, k] = V[(player, dealer, ace)]
	plot_surface(X, Y, values_to_plot[:,:,0].numpy(), "Функция ценности без играющего туза")
	plot_surface(X, Y, values_to_plot[:,:,1].numpy(), "Функция ценности с играющим туза")

def simulate_episode(env , policy):
	state = env.reset()
	is_done = False
	while not is_done:
		action = policy[state]
		state, reward, is_done, info = env.step(action)
	return reward

def eval_policy(env, optimal_policy, optimal_Q):
	n_episode = 100000
	n_win_optimal = 0
	n_lose_optimal = 0
	for _ in range(n_episode):
		reward = simulate_episode(env, optimal_policy)
		if reward == 1:
			n_win_optimal += 1
		elif reward == -1:
			n_lose_optimal += 1

	print("Вероятность выигрыша при оптимальной стратегии:", n_win_optimal/n_episode)
	print("Вероятность проигрыша при оптимальной стратегии", n_lose_optimal/n_episode)

	optimal_value = defaultdict(float)
	for state, action_values in optimal_Q.items():
		optimal_value[state] = torch.max(action_values).item()
	plot_blackjack_value(optimal_value)