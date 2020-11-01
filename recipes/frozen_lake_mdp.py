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
	total_reward = 0
	is_done = False
	while not is_done:
		action = policy[state].item()
		state, reward, is_done, info = env.step(action)
		total_reward += reward
	return total_reward

def eval_policy(env, policy, n_episode=1000):
	total_rewards = []
	for episode in range(n_episode):
		total_reward = run_episode(env, policy)
		total_rewards.append(total_reward)
	return sum(total_rewards) / n_episode

def random_search():
	i = 0
	while True:
		random_policy = torch.randint(high=n_action, size=(n_state,))
		total_reward = run_episode(env, random_policy)
		if total_reward == 1:
			best_policy = random_policy
			print(f"Found working policy on {i} episode")
			break
		i += 1

	print(best_policy)

	avg_reward = eval_policy(env, best_policy)
	print(f"Среднее полное вознаграждение при случайной стратегии: {avg_reward}")

##################

def value_iteration(env, gamma, threshold):
	"""
	Имитирует заданную окружающую среду, применяя алгоритм итерации по ценности
	@return: ценности состояний для оптимальной стратегии
	"""
	V = torch.zeros(n_state)
	while True:
		V_temp = torch.empty(n_state)
		for state in range(n_state):
			v_actions = torch.zeros(n_action)
			for action in range(n_action):
				for trans_prob, new_state, reward, _ in env.env.P[state][action]:
					v_actions[action] += trans_prob * (reward + gamma * V[new_state])
			V_temp[state] = torch.max(v_actions)
		max_delta = torch.max(torch.abs(V - V_temp))
		V = V_temp.clone()
		if max_delta <= threshold:
			break
	return V

def extract_optimal_policy(env, V_optimal, gamma):
	"""
	Строит оптимальную стратегию, соответствующую оптимальным ценностям
	@param V_optimal: оптимальные ценности
	@return: оптимальная стратегия
	"""
	optimal_policy = torch.zeros(n_state)
	for state in range(n_state):
		v_actions = torch.zeros(n_action)
		for action in range(n_action):
			for trans_prob, new_state, reward, _ in env.env.P[state][action]:
				v_actions[action] += trans_prob * (reward + gamma*V_optimal[new_state])
		optimal_policy[state] = torch.argmax(v_actions)
	return optimal_policy


def mdp_value_based():
	gamma = 0.99
	threshold = 0.0001
	V_optimal = value_iteration(env, gamma, threshold)
	print(f"Оптимальные ценности:\n{V_optimal}")
	optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
	print(f"Оптимальная стратегия:\n{optimal_policy}")
	avg_reward = eval_policy(env, optimal_policy)
	print(f"Среднее полное вознаграждение при оптимальной стратегии: {avg_reward}")

def mdp_gamma_analysis():
	gammas = [0., 0.2, 0.4, 0.6, 0.8, 0.99, 1.]
	threshold = 0.0001
	avg_reward_gamma = []
	for gamma in tqdm(gammas):
		V_optimal = value_iteration(env, gamma, threshold)
		optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
		avg_reward = eval_policy(env, optimal_policy, n_episode=10000)
		avg_reward_gamma.append(avg_reward)

	plt.plot(gammas, avg_reward_gamma)
	plt.title("Зависимость частоты успеха от коэффициента обесценивания")
	plt.xlabel("Коэффициент обесценивания")
	plt.ylabel("Средняя частота успехов")
	plt.show()

##################

def policy_estimation(env, policy, gamma, threshold):
	"""
	Выполняет оценивание стратегии
	похожа value_iteration только с заданым policy
	@param policy: матрица стратегии, содержащая вероятности действий в каждом состоянии
	@return: ценности при следовании заданной стратегии
	"""
	V = torch.zeros(n_state)
	while True:
		V_temp = torch.zeros(n_state)
		for state in range(n_state):
			action = policy[state].item()
			for trans_prob, new_state, reward, _ in env.env.P[state][action]:
				V_temp[state] += trans_prob * (reward + gamma * V[new_state])
		max_delta = torch.max(torch.abs(V - V_temp))
		V = V_temp.clone()
		if max_delta <= threshold:
			break
	return V

def policy_improvement(env, V, gamma):
	"""
	Улучшает стратегию на основе ценностей
	@param V: ценности
	@return: стратегия
	"""
	policy = torch. zeros(n_state)
	for state in range(n_state):
		v_actions = torch.zeros(n_action)
		for action in range(n_action):
			for trans_prob, new_state, reward, _ in env.env.P[state][action]:
				v_actions[action] += trans_prob * (reward + gamma*V[new_state])
		policy[state] = torch.argmax(v_actions)
	return policy

def policy_iteration(env, gamma, threshold):
	"""
	Имитирует заданную среду с помощью алгоритма итерации по стратегиям
	@return: оптимальные ценности и оптимальная стратегия для данной окружающей среды
	"""
	policy = torch.randint(high = n_action, size=(n_state,)).float()
	while True:
		V = policy_estimation(env, policy, gamma, threshold)
		policy_improved = policy_improvement(env, V, gamma)
		if torch.equal(policy_improved, policy):
			return V, policy_improved
		policy = policy_improved


def mdp_policy_base():
	gamma = 0.99
	threshold = 0.0001
	V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)
	print(f"Optimal value:\n{V_optimal}")
	print(f"Optimal policy:\n{optimal_policy}")
	avg_reward = eval_policy(env, optimal_policy)
	print(f"Среднее полное вознаграждение при оптимальной стратегии: {avg_reward}")

if __name__ == "__main__":
	# random_search()
	# mdp_value_based()
	# mdp_gamma_analysis()
	mdp_policy_base()

