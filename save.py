from datetime import datetime
from positions import get_all_positions
import imageio
import numpy
import os
import retro


def main():
    path = os.path.join(os.getcwd(), "GIFs")
    if not os.path.exists(path):
        os.mkdir(path)
    positions = get_all_positions()
    environment = retro.make(game="DonkeyKong-Nes")
    episodes = 2
    for episode in range(1, episodes + 1):
        images = []
        best_x = positions["start"][0]['x']
        best_y = positions["start"][0]['y']
        environment.reset()
        score = 0
        done = False
        while not done:
            state, reward, done, information = environment.step(environment.action_space.sample())
            score += reward
            images += [environment.render(mode="rgb_array")]
            if 0 < information['y'] < best_y:
                best_x = information['x']
                best_y = information['y']
        imageio.mimsave(os.path.join(path, str(best_y) + ' ' + str(best_x) + ' ' + str(int(score)) + ' ' +
                                     datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".gif"),
                        [numpy.array(image) for image in images], fps=30)
    environment.close()


if __name__ == '__main__':
    main()
