from collections import deque

import cv2
import numpy as np
import random
import torch


class ReplayMemory:
    def __init__(self, device, maximum_size=1_000_000):
        self.device = device
        self.buffer = deque(maxlen=maximum_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, np.array([done]))
        self.buffer.append(experience)

    def sample(self, batch_size=256):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return {"state": torch.FloatTensor(state_batch).to(self.device),
                "action": torch.FloatTensor(action_batch).to(self.device),
                "reward": torch.FloatTensor(reward_batch).to(self.device),
                "next_state": torch.FloatTensor(next_state_batch).to(self.device),
                "done": torch.FloatTensor(done_batch).to(self.device)}

    def __len__(self):
        return len(self.buffer)


def downscale(state, x, y):
    gray_image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    max_h, max_w = gray_image.shape
    h = 50
    w = 75
    y = y if y > h else h
    y = y if y + h < max_h else max_h - h
    x = x if x > w else w
    x = x if x + w < max_w else max_w - w
    frame = gray_image[y - h:y + h, x - w:x + w]
    height = int(frame.shape[0] / 4)
    width = int(frame.shape[1] / 4)
    frame = cv2.resize(frame, (width, height))
    return np.array([cv2.resize(frame, (width, height))])
