import time
import matplotlib.pyplot as plt
import torch

def show_result(env, estimator, obs2state=None, action_postprocess=None):
    state = env.reset()

    is_done = False
    while not is_done:
        if obs2state:
            state = obs2state(state)
        try:
            action = estimator.get_action(state)[0]
        except:
            q_values = estimator.predict(state)
            action = torch.argmax(q_values).item()
        if action_postprocess:
            action = action_postprocess(action)
        state, reward, is_done, _ = env.step(action)
        env.render()
    time.sleep(2)

def plot_total_reward_episoed(total_reward_episode):
	plt.plot(total_reward_episode)
	plt.title('Episode reward over time')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.show()
