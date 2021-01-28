import numpy as np
from agent.runner import Runner


class SarsaRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = self.opt.gamma  # time discount
        self.reward = self.opt.reward  # user custom setting.
        self.endpoint = self.opt.end
        self.decay = self.opt.decay
        self.Q = np.random.rand(self.ymax, self.xmax, 4)
        if self.opt.lr is None:
            self.lr = 0.1  # learning rate
        else:
            self.lr = self.opt.lr
        if self.opt.epsilon is None:
            self.epsilon = 0.5
        else:
            self.epsilon = self.opt.epsilon

    def init(self):
        self.theta_blocking()
        self.Q *= self.theta  # Wall blocking
        self.location_set()

    def location_set(self):
        super().location_set()
        self.action_init()

    def action_init(self):
        self.p = self.convert(self.theta)  # action probability
        self.act = self.next_action(self.p, [self.loc_x, self.loc_y])  # draw an action
        self.action.append(np.nan)

    def next_action(self, probability, state):
        """
        :param probability:
        :param state:
        :return:
        """
        x, y = state
        moves = sorted([self.up, self.down, self.left, self.right])
        if np.random.rand() < self.epsilon:
            next_direction = np.random.choice(moves, p=probability[y, x, :])
        else:
            next_direction = moves[np.nanargmax(self.Q[y, x, :])]
        return next_direction

    def update(self, next_state):
        nx, ny = next_state
        cx, cy = self.loc_x, self.loc_y
        if next_state == self.endpoint:  # complete an episode
            self.Q[cy, cx, self.act] = self.Q[cy, cx, self.act] \
                                       + self.lr * (self.reward - self.Q[cy, cx, self.act])
            self.move(nx, ny)
            self.epsilon *= self.decay  # policy stabilization (becoming greedy)
        else:  # not endpoint case
            next_act = self.next_action(self.p, next_state)
            self.Q[cy, cx, self.act] = self.Q[cy, cx, self.act] \
                                       + self.lr * ((1 - self.reward) +
                                                    self.gamma * self.Q[ny, nx, next_act] - self.Q[cy, cx, self.act])  # On policy
            self.move(nx, ny)
            self.act = next_act

    def run(self):
        self.action[-1] = self.act
        s_next = self.next_state(self.act)
        self.state_history.append(s_next)
        self.action.append(np.nan)
        self.update(s_next)

    def show_Q(self):
        for i in range(self.ymax):
            for j in range(self.xmax):
                print(self.Q[i][j], end=' ')
            print()
