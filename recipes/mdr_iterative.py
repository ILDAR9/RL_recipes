# Markov decision process MDP
# Марковский процесс принятия решения МППР

import torch
import matplotlib.pyplot as plt

def policy_evaluation(policy, trans_matrix, rewards, gamma, threshold):
	"""
	Оцениваем стратегию, используется только для предсказания результатовстратегии
	@param policy: матрица, оценивающая вероятности выбора дейсивий в каждом состоянии
	@param rewards: вознаграждение в каждом состоянии
	@param threshold: оценивание прекращается как только изменение всех состояний оказывается меньше порога
	@return ценность всех состояний при следовании данной стратегии
	"""
	n_state = policy.shape[0]
	V = torch.zeros(n_state)
	V_his = [V]
	while True:
		V_temp = torch.zeros(n_state)
		for state, actions in enumerate(policy):
			for action, action_prob in enumerate(actions):
				V_temp[state] += action_prob * (R[state] + gamma*torch.dot(trans_matrix[state, action], V))
		max_delta = torch.max(torch.abs(V - V_temp))
		V = V_temp.clone()
		V_his.append(V)
		if max_delta <= threshold:
			break
	return V, V_his

T = torch.tensor([[[0.8, 0.1, 0.1],
				   [0.1, 0.6, 0.3]],
				  [[0.7, 0.2, 0.1],
				   [0.1, 0.8, 0.1]],
				  [[0.6, 0.2, 0.2],
				    [0.1, 0.4, 0.5]]])

R = torch.tensor([1., 0., -1.])
gamma = 0.5
threshold = 0.0001
policy_optimal = torch.tensor([[1., 0.],
							   [1., 0.],
							   [1., 0.]])

V, V_hist = policy_evaluation(policy_optimal, T, R, gamma, threshold)
print(f"Функция ценности при оптимальной стратегии:\n{V}")

sel = lambda i: [v[i] for v in V_hist]
lines = [plt.plot(sel(i))[0] for i in range(3)]
plt.title(f"Оптимальная стратегия при game = {gamma}")
plt.xlabel("Итерация")
plt.ylabel("Ценности состояний")
plt.legend(lines,
			["State s0",
			 "State s1",
			 "State s2"], loc="upper left")
plt.show()