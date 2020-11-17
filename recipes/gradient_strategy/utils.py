import time
import matplotlib.pyplot as plt
import torch

def show_resut(env, estimator):
    state = env.reset()
    is_done = False
    while not is_done:
        q_values = estimator.predict(state)
        action = torch.argmax(q_values).item()
        state, reward, is_done, _ = env.step(action)
        env.render()
    time.sleep(3)

def show_result_action(env, estimator):
    state = env.reset()
    is_done = False
    while not is_done:
        action = estimator.get_action(state)[0]
        state, reward, is_done, _ = env.step(action)
        env.render()
    time.sleep(3)

def plot_total_reward_episoed(total_reward_episode):
	plt.plot(total_reward_episode)
	plt.title('Episode reward over time')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.show()
