from datetime import datetime
from helper import ReplayMemory
from helper import downscale
from positions import get_all_positions

import copy
import cv2
import imageio
import keyboard
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(in_features=4_930, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=4)

    def forward(self, state):
        output = f.relu(self.conv1(state))
        output = f.relu(self.conv2(output))
        output = f.relu(self.fc1(output.view(output.size()[0], -1)))
        output = f.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class DQNAgent(object):
    def __init__(self, device):
        self.device = device
        self.env = None

        self.model = QModule().to(self.device)
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.03)
        self.loss = nn.SmoothL1Loss()

        self.BATCH_SIZE = 10
        self.REPLAY_MEMORY_SIZE = 1_000_000
        self.replay_memory = ReplayMemory(self.device, maximum_size=self.REPLAY_MEMORY_SIZE)

        self.EPS_MIN = 0.01
        self.EPS_EP = 10
        self.GAMMA = 0.99
        self.TAU = 0.005

        self.explore = False

    def set(self, env):
        self.env = env

    def memorize(self, current_state, action, reward, next_state, done):
        self.replay_memory.push(current_state, action, reward, next_state, done)

    def update_target(self):
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
            next_q = self.target(next_states)
            predicted_action = torch.argmax(self.model(next_states), dim=1)

        target_q = next_q.gather(1, predicted_action.view(1, self.BATCH_SIZE))[0]
        y = rewards.T[0] + self.GAMMA * (1 - dones.T[0]) * target_q

        actions = torch.LongTensor(get_actions_number(actions)).to(self.device)
        current_q = self.model(current_states).gather(1, actions.view(1, self.BATCH_SIZE))[0]

        self.optimizer.zero_grad()
        loss = self.loss(current_q, y)
        loss.backward()
        self.optimizer.step()

        self.update_target()

    def act(self, state, episode):
        if self.explore:
            exploration = max((self.EPS_MIN - 1) / self.EPS_EP * episode + 1, self.EPS_MIN)
            if random.random() < exploration:
                return get_action(random.randint(0, 3))

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
    up = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    if action_number == 0:
        return jump
    if action_number == 1:
        return right
    if action_number == 2:
        return left
    if action_number == 3:
        return up


def get_actions_number(actions):
    actions_number = []
    for action in actions:
        action = action.detach().cpu().numpy()[::-1]
        index = np.argmax(action)
        if action.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0]:
            index = 3
        actions_number += [index]
    return np.array(actions_number)


def compute_ladder_distance(x, y, frame):
    copy_frame = frame.copy()
    copy_frame = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2GRAY)
    frame_x = copy_frame.shape[1]
    frame_y = copy_frame.shape[0]
    divide = 4
    copy_frame = cv2.resize(copy_frame, (int(frame_y / divide), int(frame_x / divide)))
    x = int(x / divide)
    y = int(y / divide)
    right = copy_frame[y + 1, x:copy_frame.shape[1]]
    left = copy_frame[y + 1, 0:x]
    left = left[::-1]
    distance_right = 9_999_999
    if (158 in right) or (159 in right):
        distance_right = max(np.argmax(right == 158), np.argmax(right == 159))
    distance_left = 9_999_999
    if 158 in left or 159 in left:
        distance_left = max(np.argmax(left == 158), np.argmax(left == 159))
    if distance_right < distance_left:
        return distance_right
    elif distance_right > distance_left:
        return -distance_left
    else:
        return 9_999_999


def compute_added_reward(prev, info, frame):
    reward = 0
    if prev is None:
        return 0
    distance = compute_ladder_distance(info['x'], info['y'], frame)
    if prev["status"] != info["status"] and info["status"] == 255:
        return -500
    if distance != 9_999_999 and info["status"] != 2:
        if distance > 0:
            if info["button"] == 1:
                reward += 100
            else:
                reward -= 200
        elif distance < 0:
            if info["button"] == 2:
                reward += 100
            else:
                reward -= 200
        elif distance == 0:
            if info["button"] == 8:
                reward += 100
    if info["status"] == 2 and info["button"] == 8:
        reward += 100
    return np.interp(reward, [-500, 500], [-1, 1])


def add_movies(agent):
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
            environment.reset()
            steps = 0
            stack_size = 1
            prev = None
            actions = []
            rewards = []
            stacked_frames = []
            while movie.step():
                steps += 1
                action = []
                for player in range(movie.players):
                    for index in range(environment.num_buttons):
                        action.append(movie.get_key(index, player))
                current_frame, reward, done, info = environment.step(action)
                reward += 2 * compute_added_reward(prev, info, current_frame)
                if done:
                    break
                current_frame = downscale(current_frame, info['x'], info['y'])
                stacked_frames.append(current_frame)
                info["action"] = action
                info["reward"] = reward
                actions.append(action)
                rewards.append(reward)
                for i in range(1, stack_size):
                    if steps - i >= 0:
                        stacked_frames[steps - i] = (np.hstack((stacked_frames[steps - i], current_frame)))
                if steps > stack_size:
                    offset = steps - (stack_size + 1)
                    average = np.array(rewards[-offset:]).mean()
                    agent.memorize(stacked_frames[offset], actions[offset], average, stacked_frames[offset + 1], done)
                if steps % 10 == 0:
                    agent.train()
                prev = info
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
    add_movies(agent)
    env = retro.make(game="DonkeyKong-Nes")
    agent.set(env)
    render = True
    stack_size = 1
    total_rewards = []
    for episode in range(1_000):
        images = []
        best_x = positions["start"][0]['x']
        best_y = positions["start"][0]['y']

        start_time = time.time()
        print("Episode:", episode + 1)

        env.reset()
        steps = 0
        done = False
        prev = None
        actions = []
        rewards = []
        stacked_frames = []
        while not done:
            if keyboard.is_pressed("f12"):
                while keyboard.is_pressed("f12"):
                    pass
                render = ~ render
            if render:
                env.render()
            if steps >= stack_size:
                offset = steps - stack_size
                action = agent.act(stacked_frames[offset], episode)
            else:
                action = get_action(random.randint(0, 3))
            current_frame, reward, done, info = env.step(action)
            reward += compute_added_reward(prev, info, current_frame)

            images += [env.render(mode="rgb_array")]
            if info["status"] != 4 and 0 < info['y'] <= best_y:
                best_x = info['x']
                best_y = info['y']

            current_frame = downscale(current_frame, info['x'], info['y'])
            stacked_frames.append(current_frame)
            info["reward"] = reward
            info["action"] = action
            actions.append(action)
            rewards.append(reward)
            for i in range(1, stack_size):
                if steps - i >= 0:
                    stacked_frames[steps - i] = (np.hstack((stacked_frames[steps - i], current_frame)))
            if steps > stack_size:
                offset = steps - (stack_size + 1)
                average = np.array(rewards[-offset:]).mean()
                agent.memorize(stacked_frames[offset], actions[offset], average, stacked_frames[offset + 1], done)
            prev = info
            if steps % 10 == 0:
                agent.train()
            steps += 1
        total_rewards += [np.array(rewards).sum()]
        print("Reward: %.3f" % np.array(rewards).sum())
        print("Steps:", steps)
        print("Duration: %.3f" % (time.time() - start_time))
        print()

        if best_y <= 205:
            imageio.mimsave(os.path.join(path, str(best_y) + ' ' + str(best_x) + ' ' +
                                         str(int(rewards[-1])) + ' ' +
                                         datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".gif"),
                            [np.array(image) for image in images], fps=30)

    env.close()


if __name__ == '__main__':
    main()
