from field.metafield import MetaField
from agent.runner import Runner
from agent.gradrunner import GradRunner
from agent.sarsarunner import SarsaRunner
from agent.qrunner import QRunner
from utils.plot import PlotGenerator, iter2dict
import pygame
import random
import os


class Maze(MetaField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 3. Maze Construction
        self.state_max = self.opt.cells
        self.start = self.opt.start
        if self.opt.end is not None:
            self.goal = self.opt.end
        else:
            self.goal = (self.state_max[0]-1, self.state_max[1]-1)
        self.n_wall = self.opt.wall

        # 4. Wall & Agent Generation
        # 4-1. Wall Generation
        self.wall_src, self.wall_dst = self.gen_wall_coord()

        # 4-2. Agent Generation
        self.agent_name = None
        self.agent = self.runner_call(self.opt.model)

    def init(self):
        super().init()
        self.agent.init()

    def get_agent_name(self):
        return self.agent_name

    def runner_call(self, command):
        if command in ['normal', 'random']:
            print('Random Selection Runner')
            self.agent_name = 'Random'
            return Runner(self.opt, [self.wall_src, self.wall_dst])
        elif command in ['grad', 'gradient']:
            print('Policy Gradient Runner')
            self.agent_name = 'Gradient'
            return GradRunner(self.opt, [self.wall_src, self.wall_dst])
        elif command in ['sarsa']:
            print('Sarsa Runner')
            self.agent_name = 'Sarsa'
            return SarsaRunner(self.opt, [self.wall_src, self.wall_dst])
        elif command in ['q', 'Q']:
            print('Q-Learning Runner')
            self.agent_name = 'Q-Learn'
            return QRunner(self.opt, [self.wall_src, self.wall_dst])
        else:
            print(f'Illegal Runner call command: {command}')
            self.exit()

    def pos_init(self, xunit, yunit):
        if xunit is None: self.xpos_unit = int(self.size[0] / self.state_max[0])
        if yunit is None: self.ypos_unit = int(self.size[1] / self.state_max[1])

    def cell(self):
        xmax, ymax = self.state_max
        idx = 0
        for y in range(ymax):
            for i, x in enumerate(range(xmax)):
                if (i + y) % 2 == 0: color = 'white'  # check type cell
                else: color = 'lightgray'
                self.draw_square(self.colorbook[color], [x, y], 1, 1, convert=True)
                self.print_text(self.font2, (x*self.xpos_unit + int(self.xpos_unit*0.35),
                                             y*self.ypos_unit + int(self.ypos_unit*0.4)),
                                self.colorbook['black'], text='S'+str(idx), aa=True)
                idx += 1
        self.print_text(self.font1, (self.start[0]*self.xpos_unit + int(self.xpos_unit*0.3),
                                     self.start[1]*self.ypos_unit + int(self.ypos_unit*0.3)),
                        self.colorbook['black'], text='START', aa=True)
        self.print_text(self.font1, (self.goal[0] * self.xpos_unit + int(self.xpos_unit * 0.35),
                                     self.goal[1] * self.ypos_unit + int(self.ypos_unit * 0.3)),
                        self.colorbook['black'], text='GOAL', aa=True)

    def wall(self, wall_src, wall_dst):
        for src, dst in zip(wall_src, wall_dst):
            self.draw_line(self.colorbook['red'], src, dst, width=5, convert=True)

    def gen_wall_coord(self):
        xs = [random.randint(1, self.state_max[0] - 1) for i in range(self.n_wall)]
        ys = [random.randint(1, self.state_max[1] - 1) for i in range(self.n_wall)]
        dx = [random.randint(-1, 1) for i in range(self.n_wall)]
        dy = [random.randint(-1, 1) for i in range(self.n_wall)]
        for i, (x, y) in enumerate(zip(dx, dy)):
            if x != 0 and y != 0:
                a = random.randint(0, 1)
                if a == 1:
                    dx[i] = 0
                else:
                    dy[i] = 0

        x_points = [x + d for x, d in zip(xs, dx)]
        y_points = [y + d for y, d in zip(ys, dy)]

        return list(zip(xs, ys)), list(zip(x_points, y_points))

    def draw_agent(self):
        tx, ty = self.agent.get_location()
        cx = tx + 0.5
        cy = ty + 0.5
        self.draw_circle(self.colorbook['darkgreen'], (cx, cy), 0.3, convert=True, radcvt='min')
        self.print_text(self.font1, (tx*self.xpos_unit + int(self.xpos_unit*0.30),
                                     ty*self.ypos_unit + int(self.ypos_unit*0.40)),
                        self.colorbook['white'], text='Agent', aa=True)

    def right_info(self, color, step, info, y_pad=50):
        x = self.state_max[0] * self.xpos_unit + 30
        y = y_pad
        self.draw_square(self.background_color, (x, y), self.x_pad, 60)
        self.print_text(self.font3, (x, y), color=color,
                        text=f'{info}: {step}',
                        aa=True)

    def train(self):
        step_record = []  # number of steps per episode
        total_steps = 0
        while True:
            if self.episode == self.opt.episode:
                break
            total_steps += self.run(total_steps, step_record)
            self.episode += 1
        pygame.quit()
        return step_record

    def run(self, total_steps, record_container):
        steps = 0
        while True:
            self.clock.tick(self.opt.speed)
            self.draw_maze(self.episode, steps, total_steps)
            self.draw_agent()
            pygame.display.flip()
            if self.opt.animation:
                self.imgsave(total_steps)
            if self.goal == self.agent.get_location():
                # complete 1 episode
                self.agent.arrival()
                break
            for event in pygame.event.get():
                self.window_escape(event)
            self.agent.run()
            steps += 1
            total_steps += 1
        print(f'steps: {steps} clear.')
        record_container.append(steps)
        return steps

    def draw_maze(self, episode, step, total_step):
        self.cell()
        self.wall(self.wall_src, self.wall_dst)
        self.right_info(self.colorbook['white'], episode+1, 'episode', 50)
        self.right_info(self.colorbook['white'], step, 'episode step', 100)
        self.right_info(self.colorbook['white'], total_step, 'total step', 150)

    def imgsave(self, total_steps):
        os.makedirs(os.path.join(self.opt.save_root, self.opt.name, 'frames'), exist_ok=True)
        rescaled = pygame.transform.scale(self.screen, self.opt.resize)
        pygame.image.save(rescaled, os.path.join(self.opt.save_root, self.opt.name, 'frames',
                                                    'step_'+str(total_steps).zfill(6)+'.jpg'))

