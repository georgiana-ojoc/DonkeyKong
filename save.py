from datetime import datetime
import imageio
import numpy
import os
import retro


def main():
    environment = retro.make(game="DonkeyKong-Nes")
    episodes = 2
    for episode in range(1, episodes + 1):
        environment.reset()
        images = []
        score = 0
        done = False
        while not done:
            state, reward, done, _ = environment.step(environment.action_space.sample())
            score += reward
            images += [environment.render(mode="rgb_array")]
        imageio.mimsave(
            os.path.join(os.getcwd(), str(score) + ' ' + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".gif"),
            [numpy.array(image) for image in images], fps=30)
    environment.close()


if __name__ == '__main__':
    main()
