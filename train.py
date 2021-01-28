from field.maze import Maze
from option import Options
from utils.plot import PlotGenerator, iter2dict
import os, glob
from PIL import Image


if __name__ == '__main__':
    opt = Options().parse()
    # Initialization
    field = Maze(opt)
    field.init()
    # Agent Training
    record = field.train()
    # Plot
    mazeplot = PlotGenerator(1, 'step record', size=(15, 10), xlabel='episode', ylabel='steps')
    mazeplot.add_set(name='Agent History', color='r')
    data = iter2dict(range(len(record)), record)
    mazeplot.add_data(data)
    mazeplot.plot()
    os.makedirs(os.path.join(opt.save_root, opt.name), exist_ok=True)
    mazeplot.save(os.path.join(opt.save_root, opt.name, 'step_record.jpg'))

    # Animation
    if opt.animation:
        directory = os.path.join(opt.save_root, opt.name, 'frames', '*')
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(directory))]
        img.save(fp=os.path.join(opt.save_root, opt.name, 'animation.gif'), format='GIF', append_images=imgs,
                 save_all=True, duration=25)