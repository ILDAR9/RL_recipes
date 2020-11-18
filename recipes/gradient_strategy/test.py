def obs2state(state):
	one_hot_state = [0] * 48
	one_hot_state[state] = 1
	return one_hot_state

a = obs2state(2)
print(a)