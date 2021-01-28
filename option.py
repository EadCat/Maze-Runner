import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # window settings
        self.parser.add_argument('--size', type=int, nargs='+', default=(800, 800), help='(X, Y) Simulation window size')
        self.parser.add_argument('--dynamic_range', type=str2bool, default=False, help='Resize the grid dynamically')
        self.parser.add_argument('--background', type=str, default='black', help='Window background color')
        self.parser.add_argument('--pad', type=int, nargs='+', default=(400, 0), help='(X, Y) window padding')
        self.parser.add_argument('--display_name', type=str, default='Maze Runner', help='Window name display')
        self.parser.add_argument('--coord_unit', '--unit', type=int, default=(None, None),
                                 help='(X, Y) Unit setting of Coordinate system, None -> default')

        # save settings
        self.parser.add_argument('--name', type=str, help='save folder name')
        self.parser.add_argument('--save_root', type=str, default='./archive', help='Save directory for some action')
        self.parser.add_argument('--animation', type=str2bool, default=False, help='Save an animation or not')
        self.parser.add_argument('--resize', type=int, default=(600, 400), help='save size of frames')

        # maze settings
        self.parser.add_argument('--cells', '-C', type=int, nargs='+', default=(5, 5), help='(X, Y) maximum maze cell')
        self.parser.add_argument('--start', type=int, nargs='+', default=(0, 0), help='(X, Y) Agent starting point')
        self.parser.add_argument('--end', type=int, nargs='+', default=None, help='(X, Y) Agent destination point')
        self.parser.add_argument('--wall', type=int, default=15, help='# of walls in the maze (random generation)')

        # training settings
        self.parser.add_argument('--model', '--agent', '-M', type=str,
                                 choices=['normal', 'random', 'grad', 'gradient', 'sarsa', 'q', 'Q'],
                                 help="Model choice: ['normal', 'random', 'grad', 'gradient', 'sarsa', 'q', 'Q']")
        self.parser.add_argument('--speed', '--tick', '-T', type=int, default=50,
                                 help='pygame.time.Clock.tick value, determine speed of simulation')
        self.parser.add_argument('--episode', '--ep', '-E', type=int, default=100, help='# of training episodes')
        self.parser.add_argument('--lr', '--learning_rate', type=float, default=None, help='Learning rate')
        self.parser.add_argument('--gamma', '-G', type=float, default=0.9, help='Value for time discount')
        self.parser.add_argument('--epsilon', '-ES', type=float, default=None,
                                 help='Exploration Prob. it will be decayed')
        self.parser.add_argument('--decay', type=float, default=0.5, help='decaying value for epsilon')
        self.parser.add_argument('--reward', '-R', type=float, default=1, help='Reinforcement reward value')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
