from datetime import datetime
from helper import ReplayMemory
from helper import downscale
from positions import get_all_positions

import copy
import imageio
import keyboard
import math
import numpy as np
import os
import random
import retro
import torch
import torch.nn as nn
import torch.nn.functional as f
import time


class QModule(nn.Module):
    def __init__(self):
        super(QModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        self.fc = nn.Linear(in_features=64 * 32, out_features=512)
        self.v = nn.Linear(in_features=512, out_features=1)
        self.a = nn.Linear(in_features=512, out_features=7)

    def forward(self, state):
        output = f.relu(self.conv1(state))
        output = f.relu(self.conv2(output))
        output = f.relu(self.conv3(output))
        output = f.relu(self.fc(output.view(output.size()[0], -1)))
        v = self.v(output)
        a = self.a(output)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DuelingQModule(nn.Module):
    def __init__(self, num_actions):
        super(DuellingQModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=math.sqrt(2.0 / (4 * 84 * 84)))
        self.conv1.bias.data.fill_(0.0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=math.sqrt(2.0 / (32 * 4 * 8 * 8)))
        self.conv2.bias.data.fill_(0.0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=math.sqrt(2.0 / (64 * 32 * 4 * 4)))
        self.conv3.bias.data.fill_(0.0)

        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        nn.init.normal_(self.fc.weight, mean=0.0, std=math.sqrt(2.0 / (64 * 64 * 3 * 3)))
        self.fc.bias.data.fill_(0.0)
        self.v = nn.Linear(in_features=512, out_features=1)
        self.a = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, states):
        output = f.relu(self.conv1(states))
        output = f.relu(self.conv2(output))
        output = f.relu(self.conv3(output))
        output = f.relu(self.fc(output.view(output.size()[0], -1)))
        v = self.v(output)
        a = self.a(output)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNAgent(object):
    def __init__(self, device):
        self.device = device
        self.env = None

        self.model = QModule().to(self.device)
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.BATCH_SIZE = 10
        self.REPLAY_MEMORY_SIZE = 1_000_000
        self.replay_memory = ReplayMemory(self.device, maximum_size=self.REPLAY_MEMORY_SIZE)

        self.EPS_MIN = 0.01
        self.EPS_EP = 100
        self.GAMMA = 0.99
        self.TAU = 0.005

        self.explore = False

    def set(self, env):
        self.env = env

    def memorize(self, current_state, action, reward, next_state, done):
        self.replay_memory.push(current_state, action, reward, next_state, done)

    def update_target(self, ):
        with torch.no_grad():
            for model_kernel, target_kernel in zip(self.model.parameters(), self.target.parameters()):
                target_kernel.copy_((1 - self.TAU) * target_kernel + self.TAU * model_kernel)

    def train(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        samples = self.replay_memory.sample(self.BATCH_SIZE)
        current_states, actions, rewards, next_states, dones = \
            samples["state"], samples["action"], samples["reward"], samples["next_state"], samples["done"]

        with torch.no_grad():
            next_action = self.model(next_states).argmax(dim=-1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_action).squeeze()

        y = rewards + (1.0 - dones) * self.GAMMA * next_q

        actions = torch.LongTensor(get_actions_number(actions)).to(self.device)
        current_q = self.model(current_states).gather(1, actions.view(1, self.BATCH_SIZE))[0]

        self.optimizer.zero_grad()
        loss = f.mse_loss(current_q, y.detach())
        loss.backward()
        self.optimizer.step()

        self.update_target()

    def act(self, state, episode):
        if self.explore:
            exploration = max((self.EPS_MIN - 1) / self.EPS_EP * episode + 1, self.EPS_MIN)
            if random.random() < exploration:
                return get_action(random.randint(0, 6))

        state = np.array([state])
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            output = self.model(state).detach().cpu().numpy()
        action = get_action(np.argmax(output))
        return action


def get_action(action_number):
    jump = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    right = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    left = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
    down = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    up = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    special = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    nothing = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    if action_number == 0:
        return jump
    if action_number == 1:
        return right
    if action_number == 2:
        return left
    if action_number == 3:
        return down
    if action_number == 4:
        return up
    if action_number == 5:
        return special
    if action_number == 6:
        return nothing


def get_actions_number(actions):
    actions_number = []
    for action in actions:
        action = action.detach().cpu().numpy()[::-1]
        index = np.argmax(action)
        if action.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1]:
            index = 5
        elif action.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0]:
            index = 6
        actions_number += [index]
    return np.array(actions_number)


def compute_added_reward(positions, prev, info, movie=False):
    if prev is None or info['x'] + info['y'] == 0:
        return 0
    if movie:
        left = 14
        right = 213
    else:
        left = 30
        right = 202
    reward = 0
    if info["status"] != 255:
        if info['x'] == prev['x'] and info['y'] == prev['y']:
            for ladder in positions["broken_ladders"]:
                if ladder['x'] - 1 <= info['x'] <= ladder['x'] + 1 and ladder['y'] - 1 <= info['y'] <= ladder['y'] + 1:
                    reward -= 50
                    break
        if info["y"] != prev['y']:
            if info["button"] == 8:
                if info['y'] < prev['y']:
                    reward += 100
            reward += 50
        if info['x'] != prev['x']:
            reward += 10
            finish = positions["finish"][0]
            if finish['x'] - 1 <= info['x'] <= finish['x'] + 1 and finish['y'] - 1 <= info['y'] <= finish['y'] + 1:
                reward += 500
                return reward
            for ladder in positions["ladders"]:
                if ladder['x'] - 1 <= info['x'] <= ladder['x'] + 1 and ladder['y'] - 1 <= info['y'] <= ladder['y'] + 1:
                    reward += 50
                    break
            for ladder in positions["hammers"]:
                if ladder['x'] - 1 <= info['x'] <= ladder['x'] + 1 and ladder['y'] - 1 <= info['y'] <= ladder['y'] + 1:
                    reward += 50
                    break
        elif (info['y'] == positions["start"][0]['y'] and info['x'] <
              positions["start"][0]['x']) or (info['x'] < left or info['x'] > right):
            reward -= 500
    elif prev["status"] != info["status"]:
        reward -= 50
    return reward


def add_movies(agent, positions):
    path = os.path.join(os.getcwd(), "movies")
    for _, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] != ".bk2":
                continue
            print(file)
            movie = retro.Movie(os.path.join(path, file))
            movie.step()
            environment = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL,
                                     players=movie.players)
            environment.initial_state = movie.get_state()
            current_frame = downscale(environment.reset(), 0, 0)
            steps = 0
            prev = None
            while movie.step():
                steps += 1
                action = []
                for player in range(movie.players):
                    for index in range(environment.num_buttons):
                        action.append(movie.get_key(index, player))
                next_frame, reward, done, info = environment.step(action)
                reward += 2 * compute_added_reward(positions, prev, info, movie=True)
                if done:
                    break
                next_frame = downscale(next_frame, info['x'], info['y'])
                info["reward"] = reward
                info["action"] = action
                agent.memorize(current_frame, action, reward, next_frame, done)
                if steps % 10 == 0:
                    agent.train()
                prev = info
                current_frame = next_frame
            environment.close()
    print()


def main():
    path = os.path.join(os.getcwd(), "GIFs")
    if not os.path.exists(path):
        os.mkdir(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print()

    positions = get_all_positions()

    agent = DQNAgent(device)
    add_movies(agent, positions)
    env = retro.make(game="DonkeyKong-Nes")
    agent.set(env)
    render = True
    total_rewards = []
    for episode in range(1_000):
        images = []
        best_x = positions["start"][0]['x']
        best_y = positions["start"][0]['y']

        start_time = time.time()
        print("Episode:", episode)

        current_state = downscale(env.reset(), 0, 0)
        steps = 0
        done = False
        prev = None
        actions = []
        rewards = []
        while not done:
            if keyboard.is_pressed("f12"):
                while keyboard.is_pressed("f12"):
                    pass
                render = ~ render
            if render:
                env.render()

            action = agent.act(current_state, episode)
            next_state, reward, done, info = env.step(action)
            reward += compute_added_reward(positions, prev, info)

            images += [env.render(mode="rgb_array")]
            if info['status'] != 4 and 0 < info['y'] <= best_y:
                best_x = info['x']
                best_y = info['y']

            next_state = downscale(next_state, info['x'], info['y'])
            info["reward"] = reward
            info["action"] = action
            actions.append(action)
            rewards.append(reward)

            agent.memorize(current_state, action, reward, next_state, done)
            if steps % 10:
                agent.train()

            steps += 1
            current_state = next_state
            prev = info

        total_rewards += [np.array(rewards).sum()]
        print("Reward: %.3f" % np.array(rewards).sum())
        print("Steps:", steps)
        print("Duration: %.3f" % (time.time() - start_time))
        print()

        if best_y <= 205:
            imageio.mimsave(os.path.join(path, str(best_y) + ' ' + str(best_x) + ' ' +
                                         str(int(total_rewards[-1])) + ' ' +
                                         datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".gif"),
                            [np.array(image) for image in images], fps=30)

    env.close()


if __name__ == '__main__':
    main()
