# Markov decision process MDR
# Марковский процесс принятия решения МППР

import torch

T = torch.tensor([[0.4, 0.6],
				   [0.8, 0.2]])
T_2 = torch.matrix_power(T, 2)
T_5 = torch.matrix_power(T, 5)
T_10 = torch.matrix_power(T, 10)
T_15 = torch.matrix_power(T, 15)
T_20 = torch.matrix_power(T, 20)

v = torch.tensor([[0.7, 0.3]])
v_1 = torch.mm(v, T)
v_2 = torch.mm(v, T_2)
v_5 = torch.mm(v, T_5)
v_10 = torch.mm(v, T_10)
v_15 = torch.mm(v, T_15)
v_20 = torch.mm(v, T_20)

print(T)
print(T_2)
print(T_5)
print(T_10)
print(T_15)
print(T_20)

print(v_1)
print(v_2)
print(v_5)
print(v_10)
print(v_15)
print(v_20)