import gym
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

n_state = env.observation_space.shape[0]
n_action = env.action_space.n

def run_episode(env, weight):
	state = env.reset()
	grads = []
	total_reward = 0
	is_done = False
	while not is_done:
		state = torch.from_numpy(state).float()
		z = torch.matmul(state, weight)
		probs = torch.nn.Softmax()(z)
		action = int(torch.bernoulli(probs[1]).item()) # only two actions in one variable

		d_softmax = torch.diag(probs) - probs.view(-1, 1) * probs
		d_log = d_softmax[action] / probs[action]
		grad = state.view(-1, 1) * d_log

		grads.append(grad)
		state, reward, is_done, _ = env.step(action)
		total_reward += reward
	return total_reward, grads

n_episode = 1000
total_rewards = []
learning_rate = 0.001
weight = torch.rand(n_state, n_action)
n_unchanged_episodes = 100

for episode in range(n_episode):
	total_reward, gradients = run_episode(env, weight)
	print(f"Эпизод {episode + 1}: {total_reward}")
	for i, gradient in enumerate(gradients):
		weight += learning_rate * gradient * (total_reward - i)
	total_rewards.append(total_reward)
	if episode >= 99 and sum(total_rewards[-100:])>195 * n_unchanged_episodes:
		break

print(f"Серднее полное вознаграждение в {n_episode} эпизодах: {sum(total_rewards)/n_episode}")

plt.plot(total_rewards)
plt.xlabel("Эпизод")
plt.ylabel("Вознаграждение")
plt.show()

n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
	total_reward, _ = run_episode(env, weight)
	print(f"Эпизод {episode + 1}: {total_reward}")
	total_rewards_eval.append(total_reward)

print(f"Серднее полное вознаграждение в {n_episode_eval} эпизодах: {sum(total_rewards_eval)/n_episode_eval}")