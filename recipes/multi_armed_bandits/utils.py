import matplotlib.pyplot as plt

def show_learning_process(action_avg_reward):
	n_action = len(action_avg_reward)
	for action in range(n_action):
		plt.plot(action_avg_reward[action])

	plt.legend(['Arm {}'.format(action) for action in range(n_action)])
	plt.xscale('log')
	plt.title('Average reward over time')
	plt.xlabel('Episode')
	plt.ylabel('Average reward')
	plt.show()