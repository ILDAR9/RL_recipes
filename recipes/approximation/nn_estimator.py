import torch
from torch.autograd import Variable
import math


class Estimator():
    def __init__(self, n_feat, n_state, n_action, n_hidden=50, lr=0.05):
        """
        @param n_hidden: if n_hidden == 0 them it will convert to Linear estimator
        """
        self.w, self.b = self.get_gaussian_wb(n_feat, n_state)
        self.n_feat = n_feat
        self.models = []
        self.optimizers = []
        self.criterion = torch.nn.MSELoss()

        for _ in range(n_action):
            if n_hidden == 0:
                model = torch.nn.Linear(n_feat, 1)
            else:
                model = torch.nn.Sequential(
                        torch.nn.Linear(n_feat, n_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, 1))
            self.models.append(model)

            if n_hidden == 0:
                optimizer = torch.optim.SGD(model.parameters(), lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr)
            self.optimizers.append(optimizer)

    def get_gaussian_wb(self, n_feat, n_state, sigma=.2):
        """
        Generate the coefficients of the feature set from Gaussian distribution
        @param n_feat: number of features
        @param n_state: number of states
        @param sigma: kernel parameter
        @return: coefficients of the features
        """
        torch.manual_seed(0)
        w = torch.randn((n_state, n_feat)) * 1.0 / sigma
        b = torch.rand(n_feat) * 2.0 * math.pi
        return w, b

    def get_feature(self, s):
        """
        Generate features based on the input state
        @param s: input state
        @return: features
        """
        features = (2.0 / self.n_feat) ** .5 * torch.cos(
            torch.matmul(torch.tensor(s).float(), self.w) + self.b)
        return features


    def update(self, s, a, y):
        """
        Update the weights for the linear estimator with the given training sample
        @param s: state
        @param a: action
        @param y: target value
        """
        features = Variable(self.get_feature(s))

        y_pred = self.models[a](features)

        loss = self.criterion(y_pred, Variable(torch.Tensor([y])))

        self.optimizers[a].zero_grad()
        loss.backward()

        self.optimizers[a].step()


    def predict(self, s):
        """
        Compute the Q values of the state using the learning model
        @param s: input state
        @return: Q values of the state
        """
        features = self.get_feature(s)
        with torch.no_grad():
            return torch.tensor([model(features) for model in self.models])

