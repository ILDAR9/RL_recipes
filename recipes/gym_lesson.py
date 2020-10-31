import gym
import time
# env = gym.make("LunarLander-v2")
# env = gym.make("SpaceInvaders-v0")
env = gym.make("CartPole-v0")
video_dir = "./cartpole_video/"
env = gym.wrappers.Monitor(env, video_dir)

env.reset()

is_done = False
while not is_done:
	action = env.action_space.sample()
	new_state, reward, is_done, info = env.step(action)
	print(info)
	env.render()

time.sleep(3)
env.render()