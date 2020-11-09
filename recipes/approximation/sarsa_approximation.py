import gym
import torch
from tqdm import tqdm
from nn_estimator import Estimator
import matplotlib.pyplot as plt
from utils import *

env = gym.envs.make("MountainCar-v0")



def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def sarsa(env, estimator, n_episode, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    """
    SARSA algorithm using Function Approximation
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_decay: epsilon decreasing factor
    """
    for episode in tqdm(range(n_episode), total=n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** episode, env.action_space.n)
        state = env.reset()
        action = policy(state)
        is_done = False
        while not is_done:
            next_state, reward, done, _ = env.step(action)
            q_values_next = estimator.predict(next_state)

            next_action = policy(next_state)

            td_target = reward + gamma * q_values_next[next_action]

            estimator.update(state, action, td_target)

            total_reward_episode[episode] += reward

            if done:
                break

            state = next_state
            action = next_action


n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_feature = 200
n_hidden = 0
lr = 0.03

estimator = Estimator(n_feature, n_state, n_action, n_hidden, lr)

n_episode = 300
total_reward_episode = [0] * n_episode

sarsa(env, estimator, n_episode, epsilon=0.1)

plot_total_reward_episoed(total_reward_episode)
show_resut(env, estimator)
