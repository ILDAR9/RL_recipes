import gym
# from cliffwalking_qlearn import q_learning
from wind_sarsa import sarsa
from utils import plot_length_reward

env = gym.make('Taxi-v3')

def find_strategy():
	gamma = 1
	alpha = 0.5
	epsilon = 0.03
	n_episode = 1000

	# optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)
	optimal_Q, optimal_policy = sarsa(env, gamma, n_episode, alpha, do_plot=True)
	print('Optimal policy:\n', optimal_policy)

def find_best_parameters():
	gamma = 1
	alpha_options = [0.4, 0.5, 0.6]
	epsilon_options = [0.1, 0.03, 0.01]
	n_episode = 500

	for alpha in alpha_options:
		for epsilon in epsilon_options:
			sarsa(env, gamma, n_episode, alpha)
			print(f'alpha: {alpha}, epsilon: {epsilon}')

if __name__ == "__main__":
	find_strategy()
	# find_best_parameters()