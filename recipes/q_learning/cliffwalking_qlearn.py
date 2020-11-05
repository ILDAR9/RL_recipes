import torch
from collections import defaultdict
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v0")

def gen_epsilon_greedy_policy(n_action, epsilon):
	def policy_function(state, Q):
		probs = torch.ones(n_action) * epsilon / n_action
		best_action = torch.argmax(Q[state]).item()
		probs[best_action] += 1.0 - epsilon
		action = torch.multinomial(probs, 1).item()
		return action
	return policy_function


def q_learning(env, gamma, n_episode, alpha):
	"""
	Obtain the optimal policy with off-policy Q-learning method
	@param alpha: learning speed
	@return: the optimal Q-function, and the optimal policy
	"""
	n_action = env.action_space.n
	Q = defaultdict(lambda: torch.zeros(n_action))
	for episode in tqdm(range(n_episode), total=n_episode):
		state = env.reset()
		is_done = False
		while not is_done:
			action = epsilon_greedy_policy(state, Q)
			next_state, reward, is_done, _ = env.step(action)
			new_action = torch.max(Q[next_state])
			td_delta = reward + gamma * new_action - Q[state][action]
			Q[state][action] += alpha * td_delta
			length_episode[episode] += 1
			total_reward_episode[episode] += reward
			if is_done:
				break
			state = next_state
	policy = {}
	for state, actions in Q.items():
		policy[state] = torch.argmax(actions).item()
	return Q, policy

gamma = 1
n_episode = 500
alpha = 0.4
epsilon = 0.1
length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode

epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)
print('Optimal policy:\n', optimal_policy)

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
