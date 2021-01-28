import pygame
import sys
import numpy as np
from typing import Iterable


class MetaField:
    def __init__(self, opt):
        pygame.init()
        self.opt = opt
        self.episode = 0
        # 1. assets registration
        self.colorbook = self.registration_color()
        self.keybook = self.registration_key()
        self.registration_font()
        # 2. window setting
        self.size = self.opt.size
        self.x_pad, self.y_pad = self.opt.pad
        self.screen = pygame.display.set_mode((self.size[0] + self.x_pad, self.size[1] + self.y_pad))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(self.opt.display_name)
        self.background_color = self.colorbook[self.opt.background]
        self.screen.fill(self.background_color)
        pygame.display.update()

    def init(self):
        self.pos_init(self.opt.coord_unit[0], self.opt.coord_unit[1])

    def registration_font(self):
        """
        https://www.cookierunfont.com/
        2019 Devsisters Corp. All Rights Reserved for "CookieRun" fonts
        """
        self.font1 = pygame.font.Font('./assets/CookieRun Bold.ttf', 15)
        self.font2 = pygame.font.Font('./assets/CookieRun Regular.ttf', 15)
        self.font3 = pygame.font.SysFont('consolas', 20)

    @staticmethod
    def registration_color():
        return {'white': np.array([255, 255, 255]),
                'black': np.array([0, 0, 0]),
                'red': np.array([255, 0, 0]),
                'green': np.array([0, 255, 0]),
                'blue': np.array([0, 0, 255]),
                'orange': np.array([255, 204, 0]),
                'azure': np.array([51, 153, 255]),
                'darkgreen': np.array([0, 102, 0]),
                'brown': np.array([153, 102, 51]),
                'purple': np.array([153, 0, 204]),
                'pink': np.array([255, 0, 255]),
                'deepred': np.array([204, 0, 0]),
                'yellow': np.array([255, 255, 0]),
                'sky': np.array([102, 204, 255]),
                'lightgray': np.array([211, 211, 211]),
                'silver': np.array([192, 192, 192])
                }

    def pos_init(self, xunit, yunit):
        """
        coordinate system unit initialization
        :param xunit: x unit
        :param yunit: y unit
        :return: Don't Return
        """
        if xunit is None:
            self.xpos_unit = 1  # unit of field coordinate
        else:
            self.xpos_unit = xunit
        if yunit is None:
            self.ypos_unit = 1  # unit of field coordinate
        else:
            self.ypos_unit = yunit

    def xcvt_unit(self, *args):
        args = list(args)
        for i in range(len(args)):
            if not isinstance(args[i], Iterable):
                args[i] = int(args[i] * self.xpos_unit)
            else:
                for j in range(len(args[i])):
                    args[i][j] = int(args[i][j] * self.xpos_unit)
        return args

    def ycvt_unit(self, *args):
        args = list(args)
        for i in range(len(args)):
            if not isinstance(args[i], Iterable):
                args[i] = int(args[i] * self.ypos_unit)
            else:
                for j in range(len(args[i])):
                    args[i][j] = int(args[i][j] * self.ypos_unit)
        return args

    def print_text(self, font, position, color, text: str, aa: bool = False):
        """
        :param font: pygame.font.Font instance
        :param position: (x, y)
        :param color: color info. [R, G, B]
        :param text: text
        :param aa: antialias
        :return: None
        """
        text_tile = font.render(text, aa, color)
        self.screen.blit(text_tile, position)

    def draw_square(self, color, position, x_width, y_height, width=0, convert=False):
        """
        :param color: color of square object
        :param position: start left-up point of square
        :param y_height:
        :param x_width:
        :param width: edge width of square
        :param convert: convert coordinate with coordinate unit or not
        :return: None
        """
        x, y = position
        if convert:
            x, x_width = self.xcvt_unit(x, x_width)
            y, y_height = self.ycvt_unit(y, y_height)
        pygame.draw.rect(self.screen, color, [x, y, x_width, y_height], width=width)

    def draw_line(self, color, start, end, width=1, convert=False):
        """
        :param color: color of polygon object
        :param start: [x1, y1]: start point
        :param end: [x2, y2]: end point
        :param width: line width
        :param convert: convert coordinate with coordinate unit or not
        :return: None
        """
        src = [0, 0]
        dst = [0, 0]
        if convert:
            src[0], dst[0] = self.xcvt_unit(start[0], end[0])
            src[1], dst[1] = self.ycvt_unit(start[1], end[1])

        pygame.draw.line(self.screen, color, src, dst, width)

    def draw_polygon(self, color, pointlist, width=0, convert=False):
        """
        :param color: color of polygon object
        :param pointlist: [[x1, y1], [x2, y2], [x3, y3], ...] list of points
        :param width: object edge width
        :param convert: convert coordinate with coordinate unit or not
        :return: None
        """
        if convert:
            for point in pointlist:
                point[0] = self.xcvt_unit(point[0])
                point[1] = self.ycvt_unit(point[1])
        pygame.draw.polygon(self.screen, color, pointlist, width)

    def draw_circle(self, color, pos, radius, width=0, convert=False, radcvt=None):
        """
        :param color: color of circle object
        :param pos: (x, y) center coordinate of circle
        :param radius: radius of circle
        :param width: edge width of circle
        :param convert: convert coordinate with coordinate unit or not
        :param radcvt: radius convert setting, options: 'x', 'y', 'min', 'max', give string
        :return: None
        """
        # process the center of circle
        if convert:
            x = self.xcvt_unit(pos[0])[0]
            y = self.ycvt_unit(pos[1])[0]
        else:
            x, y = pos
        # process radius
        if radcvt is not None:
            if radcvt == 'x':
                radius = self.xcvt_unit(radius)
            elif radcvt == 'y':
                radius = self.ycvt_unit(radius)
            elif radcvt == 'min':
                if self.xpos_unit > self.ypos_unit:
                    radius = self.ycvt_unit(radius)
                else: radius = self.xcvt_unit(radius)
            elif radcvt == 'max':
                if self.xpos_unit > self.ypos_unit:
                    radius = self.xcvt_unit(radius)
                else: radius = self.ycvt_unit(radius)
            else:
                import sys
                print('illegal circle radius convert command.')
                sys.exit()
            radius = radius[0]
        # draw
        pygame.draw.circle(self.screen, color, (x, y), radius, width)

    def window_escape(self, event):
        """
        ESC click action: exit
        :param event: keyboard input event
        :return: None
        """
        if self.keydown(event, self.keybook['ESC']):
            self.exit()

    @staticmethod
    def exit():
        pygame.quit()
        sys.exit()

    @staticmethod
    def keydown(event, target):
        if event.type == pygame.KEYDOWN:
            if event.key == target:
                return True
        return False

    @staticmethod
    def registration_key():
        return {'ENTER': pygame.K_KP_ENTER,
                'ESC': pygame.K_ESCAPE,
                'SPACE': pygame.K_SPACE,
                'ZERO': pygame.K_0,
                'ONE': pygame.K_1,
                'TWO': pygame.K_2,
                'THREE': pygame.K_3,
                'FOUR': pygame.K_4,
                'FIVE': pygame.K_5,
                'W': pygame.K_w,
                'A': pygame.K_a,
                'S': pygame.K_s,
                'D': pygame.K_d,
                'UP': pygame.K_UP,
                'DOWN': pygame.K_DOWN,
                'LEFT': pygame.K_LEFT,
                'RIGHT': pygame.K_RIGHT
                }
