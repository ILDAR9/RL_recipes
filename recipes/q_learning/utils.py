import matplotlib.pyplot as plt
import torch

def plot_length_reward(length_episode, total_reward_episode):
	plt.plot(length_episode)
	plt.title('Episode length over time')
	plt.xlabel('Episode')
	plt.ylabel('Length')
	plt.show()

	plt.plot(total_reward_episode)
	plt.title('Episode reward over time')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.show()

def gen_epsilon_greedy_policy(n_action, epsilon):
	def policy_function(state, Q):
		probs = torch.ones(n_action) * epsilon / n_action
		best_action = torch.argmax(Q[state]).item()
		probs[best_action] += 1.0 - epsilon
		action = torch.multinomial(probs, 1).item()
		return action
	return policy_function

def show_metrics(length_episode, total_reward_episode):
	n_episode = len(length_episode)
	reward_per_step = [reward/(float(step) + 0.000000001) for reward, step in zip(total_reward_episode, length_episode)]
	print(f'Average reward over {n_episode} episodes: {sum(total_reward_episode) / n_episode}')
	print(f'Average length over {n_episode} episodes: {sum(length_episode) / n_episode}')
	print(f"Average reward per step over {n_episode} episodes: {sum(reward_per_step) / n_episode}")