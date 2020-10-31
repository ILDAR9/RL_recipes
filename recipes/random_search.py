import gym
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.reset()


def run_episode(env, weight, is_show = False):
	state = env.reset()
	total_reward = 0
	is_done = False
	while not is_done:
		state = torch.from_numpy(state).float()
		action = torch.argmax(torch.matmul(state, weight))
		state, reward, is_done, _ = env.step(action.item())
		total_reward += reward
		if is_show:
			env.render()
	return total_reward

n_episode = 1000
best_total_reward = 0
best_weight = None
total_rewards = []

n_state = env.observation_space.shape[0]
n_action = env.action_space.n

for episode in tqdm(range(n_episode), total=n_episode):
	weight = torch.rand(n_state, n_action)
	total_reward = run_episode(env, weight)
	# print(f"Эпизод {episode+1}: {total_reward}")
	if total_reward > best_total_reward:
		best_weight = weight
		best_total_reward = total_reward
	total_rewards.append(total_reward)

print(f"Среднее полное вознаграждение в {n_episode} эпизодах: " \
	  f"{sum(total_rewards) / n_episode}")

n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
	total_reward = run_episode(env, best_weight)
	print(f"Эпизод {episode + 1}: {total_reward}")
	total_rewards_eval.append(total_reward)

print(f"Среднее полное вознаграждение в {n_episode_eval} эпизодах: " \
	  f"{sum(total_rewards_eval) / n_episode_eval}")

# run_episode(env, best_weight, is_show=True)
# time.sleep(3)

plt.plot(total_rewards)
plt.xlabel("Эпизод")
plt.ylabel("Вознаграждение")
plt.show()

