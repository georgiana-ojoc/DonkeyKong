import cv2
import neat
import numpy as np
import pickle
import retro

env = retro.make(game="DonkeyKong-Nes")


def distance_to_finish(x, y):
    jumpman_position = np.array((x, y))
    finish = np.array((145, 33))

    return np.linalg.norm(jumpman_position - finish)


def calc_reward(info):
    reward = 0
    if info['y'] != 0 and info["min_y"] > info['y']:
        reward += (info["min_y"] - info['y']) * 5
    return (1.0 / (distance_to_finish(info['x'], info['y']) + 1e-8)) * 10 + reward


def downscale(state, x, y):
    gray_image = np.array(cv2.cvtColor(state, cv2.COLOR_BGR2GRAY))
    max_h, max_w = gray_image.shape
    h = 50
    w = 75
    y = y if y > h else h
    y = y if y + h < max_h else max_h - h
    x = x if x > w else w
    x = x if x + w < max_w else max_w - w
    frame = gray_image[y - h:y + h, x - w:x + w]
    height = int(frame.shape[0] / 6)
    width = int(frame.shape[1] / 6)
    frame = cv2.resize(frame, (width, height))
    return frame


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        current_frame = env.reset()
        net = neat.nn.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        current_fitness = 0
        frame = 0
        counter = 0

        done = False
        min_y = 204
        prevs = []
        while not done:
            if frame % 15 == 0:
                env.render()
            frame += 1

            if len(prevs) != 0:
                current_frame = downscale(current_frame, prevs[0]['x'], prevs[0]['y'])
            else:
                current_frame = downscale(current_frame, 0, 0)

            flatten_image = np.ndarray.flatten(current_frame)
            neural_network_output = net.activate(flatten_image)
            current_frame, reward, done, info = env.step(neural_network_output)
            if len(prevs) > 0:
                if prevs[0]["status"] != info["status"] and info["status"] == 255:
                    done = True

            info["min_y"] = min_y
            reward += calc_reward(info)
            if info['y'] != 0 and info['y'] < min_y:
                min_y = info['y']
            prevs.insert(0, info)
            current_fitness += reward
            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print("%d %.6f" % (genome_id, current_fitness))

            genome.fitness = current_fitness


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config-feedforward")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    winner = p.run(eval_genomes)
    with open("winner.pkl", "wb") as output:
        pickle.dump(winner, output, 1)


if __name__ == '__main__':
    main()
