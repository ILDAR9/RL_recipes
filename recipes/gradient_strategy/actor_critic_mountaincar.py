import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import numpy as np
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings("ignore")

env = gym.make('MountainCarContinuous-v0')


class ActorCriticModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(ActorCriticModel, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.mu = nn.Linear(n_hidden, n_output)
        self.sigma = nn.Linear(n_hidden, n_output)
        self.value = nn.Linear(n_hidden, 1)
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5
        dist = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        value = self.value(x)
        return dist, value


class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden, lr=0.001):
        self.model = ActorCriticModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)


    def update(self, returns, log_probs, state_values):
        """
        Update the weights of the Actor Critic network given the training samples
        @param returns: return (cumulative rewards) for each step in an episode
        @param log_probs: log probability for each step
        @param state_values: state-value for each step
        """
        loss = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = - log_prob * advantage

            value_loss = F.smooth_l1_loss(value, Gt)

            loss += policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, s):
        """
        Compute the output using the continuous Actor Critic model
        @param s: input state
        @return: Gaussian distribution, state_value
        """
        self.model.training = False
        return self.model(torch.Tensor(s))

    def get_action(self, s):
        """
        Estimate the policy and sample an action, compute its log probability
        @param s: input state
        @return: the selected action, log probability, predicted state-value
        """
        dist, state_value = self.predict(s)
        action = dist.sample().numpy()
        log_prob = dist.log_prob(action[0])
        return action, log_prob, state_value




def actor_critic(env, estimator, n_episode, gamma=1.0):
    """
    continuous Actor Critic algorithm
    @param env: Gym environment
    @param estimator: policy network
    @param n_episode: number of episodes
    @param gamma: the discount factor
    """
    for episode in tqdm(range(n_episode), total=n_episode):
        log_probs = []
        rewards = []
        state_values = []
        state = env.reset()

        while True:
            state = scale_state(state)
            action, log_prob, state_value = estimator.get_action(state)
            action = action.clip(env.action_space.low[0],
                                 env.action_space.high[0])
            next_state, reward, is_done, _ = env.step(action)

            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)

            if is_done:
                returns = []

                Gt = 0
                pw = 0

                for reward in rewards[::-1]:

                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)


                estimator.update(returns, log_probs, state_values)

                break

            state = next_state


state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)


def scale_state(state):
    scaled = scaler.transform([state])
    return scaled[0]


n_state = env.observation_space.shape[0]
n_action = 1
n_hidden = 128
lr = 0.0003
policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)


n_episode = 200
gamma = 0.9
total_reward_episode = [0] * n_episode

actor_critic(env, policy_net, n_episode, gamma)

action_postprocess = lambda action:  action.clip(env.action_space.low[0], env.action_space.high[0])
plot_total_reward_episoed(total_reward_episode)
show_result(env, policy_net, obs2state=scale_state, action_postprocess=action_postprocess)
