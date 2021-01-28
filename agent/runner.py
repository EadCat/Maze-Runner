import numpy as np


class Runner:
    def __init__(self, opt, wall_pts):
        self.opt = opt
        self.ep = 0  # episode
        # direction definition
        self.up = 1
        self.down = 2
        self.left = 0
        self.right = 3
        # field setting
        self.xmax, self.ymax = opt.cells
        self.wall_src, self.wall_dst = wall_pts
        self.start_x, self.start_y = opt.start
        # [left, up, down, right]
        self.theta = np.ones(shape=(self.ymax, self.xmax, 4))
        # get to the starting point, making record container
        self.state_history = []
        self.action = []

    def init(self):
        self.theta_blocking()
        self.location_set()

    def clear(self):
        self.state_history.clear()
        self.action.clear()

    def theta_blocking(self):
        """
        blocking all wall direction thetas
        use with a class initializer
        :return: None
        """
        # blocking
        self.theta[0, :, self.up] = np.nan  # block upper border
        self.theta[self.ymax - 1, :, self.down] = np.nan  # block lower border
        self.theta[:, 0, self.left] = np.nan  # block left border
        self.theta[:, self.xmax - 1, self.right] = np.nan  # block right border

        for (xs, ys), (xe, ye) in zip(self.wall_src, self.wall_dst):
            if xs == xe and ys == ye:
                pass
            elif xs == xe:
                ycall = min(ys, ye)
                self.theta[ycall, xs, self.left] = np.nan
                self.theta[ycall, xs - 1, self.right] = np.nan
            elif ys == ye:
                xcall = min(xs, xe)
                self.theta[ys, xcall, self.up] = np.nan
                self.theta[ys - 1, xcall, self.down] = np.nan
            else:
                print('impossible')

    def location_set(self):
        """
        agent location reset
        :return: None
        """
        self.state_history.append((self.start_x, self.start_y))
        self.loc_x, self.loc_y = self.start_x, self.start_y
        self.ep += 1

    def run(self):
        p = self.convert(self.theta)
        act = self.next_action(p)
        x, y = self.next_state(act)
        self.move(x, y)
        self.state_history.append((x, y))
        self.action.append(act)

    @staticmethod
    def convert(theta):
        """
        theta -> probability (not softmax)
        :param theta:
        :return: move probability
        """
        # [y][x][4]
        move_prob = np.zeros(shape=theta.shape)
        ys, xs, _ = theta.shape
        for y in range(ys):
            for x in range(xs):
                move_prob[y, x, :] = theta[y, x, :] / np.nansum(theta[y, x, :])
        move_prob = np.nan_to_num(move_prob)
        return move_prob

    def move(self, x, y):
        """
        change "real agent location"
        :return: None
        """
        self.loc_x, self.loc_y = x, y

    def next_action(self, probability):
        """
        choose next action probabilistically, but "not change" real location
        :param probability: theta->probability result value
        :return: next action
        """
        moves = sorted([self.up, self.down, self.left, self.right])
        next_direction = np.random.choice(moves, p=probability[self.loc_y][self.loc_x][:])
        return next_direction

    def next_state(self, action):
        """
        estimate next location, but "not change" real location
        :param action: an action judge from self.next_action(p)
        :return: estimated location by moving with input action
        """
        current_x = self.loc_x
        current_y = self.loc_y
        if action == self.up:
            current_y -= 1
        elif action == self.down:
            current_y += 1
        elif action == self.left:
            current_x -= 1
        elif action == self.right:
            current_x += 1
        else:
            import sys
            print('Illegal direction estimation.')
            sys.exit()
        return [current_x, current_y]

    def arrival(self):
        """
        an action when arriving
        :param goal: goal location (x, y)
        :return: None
        """
        self.action.append(np.nan)
        self.location_set()

    def __len__(self):
        return len(self.state_history)

    def get_location(self):
        return self.loc_x, self.loc_y

    def get_history(self):
        return self.state_history

    def get_theta(self):
        return self.theta

    def get_prob(self):
        return self.convert(self.theta)

    def get_delta(self):
        raise NotImplementedError

    def show_theta(self):
        for i in range(self.ymax):
            for j in range(self.xmax):
                print(self.theta[i][j], end=' ')
            print()

    def show_prob(self):
        prob = self.get_prob()
        for i in range(self.ymax):
            for j in range(self.xmax):
                print(prob[i][j], end=' ')
            print()