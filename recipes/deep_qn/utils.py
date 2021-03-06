import time
import matplotlib.pyplot as plt
import torch

def show_resut(env, estimator, obs2state=None):
    state = env.reset()
    if obs2state:
        state = obs2state(state)
    is_done = False
    while not is_done:
        q_values = estimator.predict(state)
        action = torch.argmax(q_values).item()
        state, reward, is_done, _ = env.step(action)
        if obs2state:
            state = obs2state(state)
        env.render()
    time.sleep(3)

def plot_total_reward_episoed(total_reward_episode):
	plt.plot(total_reward_episode)
	plt.title('Episode reward over time')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.show()
