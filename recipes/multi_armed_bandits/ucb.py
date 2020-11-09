import torch
from tqdm import tqdm
from multi_armed_bandit import BanditEnv
from utils import show_learning_process

def upper_confidence_bound(Q, action_count, t):
    ucb = torch.sqrt((2 * torch.log(torch.tensor(float(t)))) / action_count) + Q
    return torch.argmax(ucb)

def ucb(bandit_env, n_episode):
    n_action = len(bandit_env.payout_list)
    action_count = torch.tensor([0. for _ in range(n_action)])
    action_total_reward = [0 for _ in range(n_action)]
    action_avg_reward = [[] for action in range(n_action)]


    Q = torch.empty(n_action)

    for episode in tqdm(range(n_episode), total=n_episode):
        action = upper_confidence_bound(Q, action_count, episode)
        reward = bandit_env.step(action)
        action_count[action] += 1
        action_total_reward[action] += reward
        Q[action] = action_total_reward[action] / action_count[action]

        for a in range(n_action):
            if action_count[a]:
                action_avg_reward[a].append(action_total_reward[a] / action_count[a])
            else:
                action_avg_reward[a].append(0)

    show_learning_process(action_avg_reward)
    print("Среднее вознаграждения", sum(action_total_reward) / n_episode)
    return Q

if __name__ == "__main__":
    bandit_payout = [0.1, 0.15, 0.3]
    bandit_reward = [4, 3, 1]
    bandit_env = BanditEnv(bandit_payout, bandit_reward)
    n_episode = 100000
    ucb(bandit_env, n_episode)