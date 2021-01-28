import numpy as np
from agent.runner import Runner


class GradRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.opt.lr is None:
            self.lr = 1
        else:
            self.lr = self.opt.lr
        if self.opt.epsilon is None:
            self.beta = 1
        else:
            self.beta = self.opt.epsilon

    def run(self):
        p = self.convert(self.theta)
        act = self.next_action(p)
        x, y = self.next_state(act)
        self.move(x, y)
        self.state_history.append((x, y))
        self.action.append(act)

    def convert(self, theta):
        """
        Convert with Softmax
        :param theta:
        :return:
        """
        ys, xs, _ = theta.shape
        move_prob = np.zeros(shape=theta.shape)
        exp_theta = np.exp(self.beta * theta)
        for y in range(ys):
            for x in range(xs):
                move_prob[y, x, :] = exp_theta[y, x, :] / np.nansum(exp_theta[y, x, :])
        move_prob = np.nan_to_num(move_prob)
        return move_prob

    def arrival(self):
        super().arrival()
        self.update(self.convert(self.theta))
        self.clear()

    def update(self, p):
        """
        registrate new theta, delta
        :param p: probability
        :return:
        """
        current_steps = len(self) - 1
        delta = np.zeros(shape=self.theta.shape)  # [y, x, 4]
        subdelta = np.zeros(shape=self.theta.shape)
        ys, xs, _ = delta.shape

        for act, state in zip(self.action, self.state_history):
            if not np.isnan(act):
                x, y = state
                delta[y, x, :] += 1
                subdelta[y, x, act] += 1

        delta_theta = (subdelta - p * delta) / current_steps

        self.theta = self.theta + self.lr * delta_theta
        self.delta = np.sum(np.abs(self.convert(self.theta)-p))

    def get_delta(self):
        return self.delta


