import torch
from windy_gridworld import WindyGridworldEnv
from collections import defaultdict
from tqdm import tqdm
from utils import plot_length_reward, gen_epsilon_greedy_policy


def sarsa(env, gamma, n_episode, alpha, do_plot=False):
    """
    Obtain the optimal policy with on-policy SARSA algorithm
    @return: the optimal Q-function, and the optimal policy
    """
    epsilon = 0.1
    epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
    length_episode = [0] * n_episode
    total_reward_episode = [0] * n_episode 

    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in tqdm(range(n_episode), total = n_episode):
        state = env.reset()
        is_done = False
        action = epsilon_greedy_policy(state, Q)
        while not is_done:
            next_state, reward, is_done, _ = env.step(action)
            next_action = epsilon_greedy_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    if do_plot:
        plot_length_reward(length_episode, total_reward_episode)
    show_metrics(length_episode, total_reward_episode)
    return Q, policy

if __name__ == "__main__":
    env = WindyGridworldEnv()
    gamma = 1
    n_episode = 500
    alpha = 0.4

    optimal_Q, optimal_policy = sarsa(env, gamma, n_episode, alpha)

    print('The optimal policy:\n', optimal_policy)
