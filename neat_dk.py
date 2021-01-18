import cv2  # For image reduction
import math
import neat
import numpy as np
import pickle
import retro

oned_image = []


def calc_reward(info, prevs):
    rew = 0
    if info['y'] != 0 and info['min_y'] > info['y']:
        rew += (info['min_y'] - info['y']) * 5
    return (1 / math.sqrt(((info['x'] - 80) ** 2) + ((info['y'] - 30) ** 2)) * 10) + rew


def downscale(state, x, y):
    grayImage = np.array(cv2.cvtColor(state, cv2.COLOR_BGR2GRAY))
    max_h, max_w = grayImage.shape
    h = 50
    w = 75
    y = y if y > h else h
    y = y if y + h < max_h else max_h - h
    x = x if x > w else w
    x = x if x + w < max_w else max_w - w
    frame = grayImage[y - h:y + h, x - w:x + w]
    height = int(frame.shape[0] / 6)
    width = int(frame.shape[1] / 6)
    frame = cv2.resize(frame, (width, height))
    # cv2.imshow("small", frame)
    # copy = cv2.resize(frame, (width*10, height*10))
    # cv2.imshow("big",copy)
    return frame


env = retro.make(game='DonkeyKong-Nes')


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()  # First image
        random_action = env.action_space.sample()
        # 20 Networks
        net = neat.nn.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0

        done = False
        prevs = []
        min_y = 204
        while not done:
            if frame % 15 == 0:
                env.render()  # Optional
            frame += 1

            if len(prevs) != 0:
                ob = downscale(ob, prevs[0]['x'], prevs[0]['y'])
            else:
                ob = downscale(ob, 0, 0)

            oned_image = np.ndarray.flatten(ob)
            neuralnet_output = net.activate(oned_image)  # Give an output for current frame from neural network
            ob, rew, done, info = env.step(neuralnet_output)  # Try given output from network in the game
            if len(prevs) > 0:
                if prevs[0]['status'] != info['status'] and info['status'] == 255:
                    done = True
            info['min_y'] = min_y
            rew += calc_reward(info, prevs)
            if info['y'] != 0 and info['y'] < min_y:
                min_y = info['y']
            prevs.insert(0, info)
            fitness_current += rew
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                # count the frames until it successful

            # Train for max 250 frames
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint('first-run-saves/neat-checkpoint-29')

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
# Save the process after each 10 frames
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open("winner.pkl", "wb") as output:
    pickle.dump(winner, output, 1)
