import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

gamma = 1 # нету обесцениваиня
threshold = 1e-10
capital_max = 100
n_state = capital_max + 1
rewards = torch.zeros(n_state)
rewards[-1] = 1

env = {'capital_max': capital_max,
	'head_prob': 0.4,
	'rewards': rewards,
	'n_state': n_state
}

def value_iteration(env, gamma, threshold):
	"""
	Решает задачу о подбрасывании монеты с помощью алгоритма итерации по ценности
	@return: ценности при следовании оптимальной стратегии для данной среды
	"""
	head_prob = env['head_prob']
	n_state = env['n_state']
	capital_max = env['capital_max']
	V = torch.zeros(n_state)
	while True:
		V_temp = torch.zeros(n_state)
		for state in range(1, capital_max):
			n_action = min(state, capital_max - state) + 1
			v_actions = torch.zeros(n_action)
			for action in range(1, n_action):
				v_actions[action] += head_prob * (rewards[state + action] + gamma*V[state + action])
				v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma*V[state - action])
			V_temp[state] = torch.max(v_actions)
		max_delta = torch.max(torch.abs(V - V_temp))
		V = V_temp.clone()
		if max_delta <= threshold:
			break
	return V

def extract_optimal_policy(env, V_optimal, gamma):
	"""
	Строит оптимальную стратегию по оптимальным ценностям
	@param V_optimal: оптимальные ценности
	@return: оптимальная стратегия
	"""
	head_prob = env['head_prob']
	n_state = env['n_state']
	capital_max = env['capital_max']
	optimal_policy = torch.zeros(capital_max).int()
	for state in range(1, capital_max):
		v_actions = torch.zeros(n_state)
		for action in range(1, min(state, capital_max - state) + 1):
			v_actions[action] += head_prob * (rewards[state + action] + gamma*V_optimal[state + action])
			v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma*V_optimal[state - action])
		optimal_policy[state] = torch.argmax(v_actions)
	return optimal_policy

def mdp_value_base():
	state_time = time()
	V_optimal = value_iteration(env, gamma, threshold)
	optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
	print(f"Для решения методом итерации по ценности понадобилось {time() - state_time:.3f} с")
	print(f"Оптимальные ценности:\n{V_optimal}")
	print(f"Оптимальная стратегия:\n{optimal_policy}")

	plt.plot(V_optimal[:100].numpy())
	plt.title("Оптимальные ценности состояний")
	plt.xlabel("Капитал")
	plt.ylabel("Ценность")
	plt.show()

###################

def policy_estimation(env, policy, gamma, threshold):
	"""
	Оценивает стратегию
	@param policy: тензор стратегии, содержащий действия, предпринимаемые в каждом состоянии
	"""
	head_prob = env['head_prob']
	n_state = env['n_state']
	capital_max = env['capital_max']
	V = torch.zeros(n_state)
	while True:
		V_temp = torch.zeros(n_state)
		for state in range(1, capital_max):
			action = policy[state].item()
			V_temp[state] += head_prob * (rewards[state + action] + gamma*V[state + action])
			V_temp[state] += (1 - head_prob) * (rewards[state - action] + gamma*V[state - action])
		max_delta = torch.max(torch.abs(V - V_temp))
		V = V_temp.clone()
		if max_delta <= threshold:
			break
	return V

def policy_improvement(env, V, gamma):
	"""
	Строит улучшенную стратегию на основе ценностей
	@param V: ценности состояний
	@return: стртегия
	"""
	head_prob = env['head_prob']
	n_state = env['n_state']
	capital_max = env['capital_max']
	policy = torch.zeros(n_state).int()
	for state in range(1, capital_max):
		n_action = min(state, capital_max - state) + 1
		v_actions = torch.zeros(n_action)
		for action in range(1, n_action):
			v_actions[action] += head_prob * (rewards[state + action] + gamma*V[state + action])
			v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma*V[state - action])
		policy[state] = torch.argmax(v_actions)
	return policy

def policy_iteration(env, gamma, threshold):
	"""
	Решает задачу о подбрасывании монеты с помощью алгоритма итерации по стратегии
	@return: оптимальные ценности и оптимальная стратегия для данной среды
	"""
	n_state = env['n_state']
	policy = torch.zeros(n_state).int()
	while True:
		V = policy_estimation(env, policy, gamma, threshold)
		policy_improved = policy_improvement(env, V, gamma)
		if torch.equal(policy_improved, policy):
			return V, policy_improved
		policy = policy_improved

def mdp_policy_based():
	state_time = time()
	V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)
	print(f"Для решения методом итерации по стратегии понадобилось {time() - state_time:.3f} с")
	print(f"Оптимальные ценности:\n{V_optimal}")
	print(f"Оптимальная стратегия:\n{optimal_policy}")

###############

def optimal_strategy(capital, policy_store = []):
	if not policy_store:
		print("Создает оптимальную стратегию")
		_, optimal_policy = policy_iteration(env, gamma, threshold)
		policy_store.append(optimal_policy)
	return policy_store[0][capital].item()

def conservative_strategy(capital):
	return 1

def random_strategy(capital):
	return torch.randint(1, capital + 1, (1,)).item()

def run_episode(head_prob, capital, policy):
	while capital > 0:
		bet = policy(capital)
		if torch.rand(1).item() < head_prob:
			capital += bet
			if capital >= 100:
				return 1
		else:
			capital -= bet
	return 0

def compare_strategies():
	capital = 50
	n_episode = 10000
	n_win_random = 0
	n_win_conservative = 0
	n_win_optimal = 0
	head_prob = env['head_prob']
	for episode in tqdm(range(n_episode), total=n_episode):
		n_win_random += run_episode(head_prob, capital, random_strategy)
		n_win_conservative += run_episode(head_prob, capital, conservative_strategy)
		n_win_optimal += run_episode(head_prob, capital, optimal_strategy)
	print(f"Средняя вероятность выигрыша при случайной стратегии: {n_win_random/n_episode}")
	print(f"Средняя вероятность выигрыша при консервативной стратегии: {n_win_conservative/n_episode}")
	print(f"Средняя вероятность выигрыша при оптимальной стратегии: {n_win_optimal/n_episode}")



if __name__ == "__main__":
	# mdp_value_base()
	# mdp_policy_based()
	compare_strategies()



