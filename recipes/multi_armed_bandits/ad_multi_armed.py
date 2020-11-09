import torch
from multi_armed_bandit import BanditEnv
from utils import show_learning_process
from ucb import ucb




if __name__ == "__main__":
    bandit_payout = [0.01, 0.015, 0.03]
    bandit_reward = [1, 1, 1]
    bandit_env = BanditEnv(bandit_payout, bandit_reward)

    n_episode = 100000
    Q = ucb(bandit_env, n_episode)
