import torch
import gym
from tqdm import tqdm
from utils import plot_length_reward, gen_epsilon_greedy_policy, show_metrics

env = gym.make('Taxi-v3')


def double_q_learning(env, gamma, n_episode, alpha):
    """
    Obtain the optimal policy with off-policy double Q-learning method
    @param gamma: discount factor
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    n_state = env.observation_space.n
    Q1 = torch.zeros(n_state, n_action)
    Q2 = torch.zeros(n_state, n_action)
    for episode in tqdm(range(n_episode), total=n_episode):
        state = env.reset()
        is_done = False
        while not is_done:
            action = epsilon_greedy_policy(state, Q1 + Q2) # just fist random action don't mind
            next_state, reward, is_done, info = env.step(action)
            if (torch.rand(1).item() < 0.5):
                best_next_action = torch.argmax(Q1[next_state])
                td_delta = reward + gamma * Q2[next_state][best_next_action] - Q1[state][action]
                Q1[state][action] += alpha * td_delta
            else:
                best_next_action = torch.argmax(Q2[next_state])
                td_delta = reward + gamma * Q1[next_state][best_next_action] - Q2[state][action]
                Q2[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
    policy = {}
    Q = Q1 + Q2
    for state in range(n_state):
        policy[state] = torch.argmax(Q[state]).item()
    return Q, policy

gamma = 1

n_episode = 3000

alpha = 0.4

epsilon = 0.1

epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)

length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode

optimal_Q, optimal_policy = double_q_learning(env, gamma, n_episode, alpha)

plot_length_reward(length_episode, total_reward_episode)
show_metrics(length_episode, total_reward_episode)
print(optimal_policy)

