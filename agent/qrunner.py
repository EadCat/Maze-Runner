import numpy as np

from agent.runner import Runner
from agent.sarsarunner import SarsaRunner


class QRunner(SarsaRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, next_state):
        nx, ny = next_state
        cx, cy = self.loc_x, self.loc_y
        if next_state == self.endpoint:
            self.Q[cy, cx, self.act] = self.Q[cy, cx, self.act] \
                                       + self.lr * (self.reward - self.Q[cy, cx, self.act])
            self.move(nx, ny)
            self.epsilon /= 2  # policy stabilization (becoming greedy)
        else:
            next_act = self.next_action(self.p, next_state)
            self.Q[cy, cx, self.act] = self.Q[cy, cx, self.act] \
                                       + self.lr * ((1 - self.reward) +
                                                    self.gamma * np.nanmax(self.Q[ny, nx, :]) - self.Q[cy, cx, self.act])
            self.move(nx, ny)
            self.act = next_act
