import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import random
import gym
import time
import retro


env = retro.make(game='DonkeyKong-Nes')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_to_raw(actions):
	raw_actions=[]
	for action in actions:
		action=action.numpy()
		raw_actions+=[int(action.dot(2**np.arange(action.size)[::-1]))]
	return raw_actions
def raw_to_action(number):
	binary = []
	for i in range(9):
		bit = number % 2
		binary.insert(0, bit)
		number = number // 2
	return binary

#aici nu cred ca mai trebuie sa umblam, deci VERIFIED
class ReplayBuffer:
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
            state_batch.append(np.transpose(state,(2,0,1)))
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(np.transpose(state,(2,0,1)))
            done_batch.append(done)

        return {"state": torch.FloatTensor(state_batch).to(self.device),
                "action": torch.FloatTensor(action_batch).to(self.device),
                "reward": torch.FloatTensor(reward_batch).to(self.device),
                "next_state": torch.FloatTensor(next_state_batch).to(self.device),
                "done": torch.FloatTensor(done_batch).to(self.device)}

    def __len__(self):
        return len(self.buffer)

#aici putem si cred ca va trebui sa ne jucam cu size-ul layerelor de convolutie
# numerele 12 si 13 sunt produse de 224/16 si 240/16 rezultate in urma pooling-ului de 4 (efectuat de doua ori)
class Q_Module(nn.Module):
	def __init__(self,state_n,action_n):
		super(Q_Module, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(4, 4)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 12*13, 100)
		self.fc2 = nn.Linear(100, 200)
		self.fc3 = nn.Linear(200, 512)

	def forward(self, state):
		output = self.pool(F.relu(self.conv1(state)))
		output = self.pool(F.relu(self.conv2(output)))
		output = output.view(-1, 16 * 12 * 13)
		output = F.relu(self.fc1(output))
		output = F.relu(self.fc2(output))
		output = self.fc3(output)
		return output


#Trebuie sa modificam actiunile de output daca dorim sa facemproblema mai mica
class DQNAgent(object):
	def __init__(self, env):

		self.env=env

		action_n=env.action_space.shape[0]
		state_n=env.observation_space.shape

		#Actor and target Actor using Adam optimizer
		self.model = Q_Module(state_n,512).to(device)
		self.target_model = copy.deepcopy(self.model)
		self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

		#Replay memory for training
		self.BATCH_SIZE = 10
		self.REPLAY_MEMORY_SIZE = 1_000_000
		self.replay_memory = ReplayBuffer(device)
		#constants
		self.EPS_MIN=0.01
		self.EPS_EP=100
		self.GAMMA = 0.99
		self.TAU=.005
	
	def memorize(self, current_state, action, reward, next_state, done):
		self.replay_memory.push(current_state, action, reward, next_state, done)

	def act(self, state, episode):
		exploration = max((self.EPS_MIN - 1) / 10 * episode + 1, self.EPS_MIN)
		if random.random() < exploration:
			return self.env.action_space.sample()
		
		
		state=np.array([np.transpose(state,(2,0,1))])
		state=torch.FloatTensor(state).to(device)
		with torch.no_grad():
			raw_action=torch.argmax(self.model( state)).detach().numpy()
		action=raw_to_action(raw_action)
		return action

	def update_target(self,):
		with torch.no_grad():
			for model_kernel, model_target_kernel in zip(self.model.parameters(), self.target_model.parameters()):
				model_target_kernel.copy_((1-self.TAU)*model_target_kernel +  self.TAU * model_kernel)

	def train(self, steps):
		if len(self.replay_memory) < self.BATCH_SIZE:
			return
		samples = self.replay_memory.sample(self.BATCH_SIZE)
		self.train_with_sample(samples,steps)
	
	def train_with_sample(self,samples,steps):

		current_state, action, reward, next_state, done = \
            samples["state"], samples["action"], samples["reward"], samples["next_state"], samples["done"]

		#Compute Y = r + γ * (1-done) * Q_target(s',a')
		with torch.no_grad():
			target_q=torch.max(self.target_model(next_state),axis=1)[0]
			Y= reward.T[0] + self.GAMMA*(1-done.T[0])* target_q

		#Compute critic loss like the sum of the mean squared error of the current q value
		raw_action=action_to_raw(action)
		current_q = torch.max(self.model(current_state),axis=1)[0]
		loss		= 	F.mse_loss(current_q,Y)
		
		#Update critic model using the previous computed loss
		self.model_optimizer.zero_grad()
		loss.backward()
		self.model_optimizer.step()

		self.update_target()


def main():
	#TRAIN PHASE
	agent=DQNAgent(env)
	rewards_per_episode=[]

	for episode in range(1_000):
		start_time = time.time()
		print("EPISODE: ", episode)
		current_state = env.reset()

		steps=0
		done = False
		rewards = []
		while not done:
			if episode % 100 == 0:
				env.render()

			action = agent.act(current_state,episode)
			#Repeta actiunile un numar de frame-uri
			for i in range(30):
				next_state, reward, done, _ = env.step(action)
				rewards += [reward]

				agent.memorize(current_state, action, reward, next_state, done)
			agent.train(steps)

			current_state = next_state
			steps+=1
		rewards_per_episode += [np.array(rewards).sum()]
		print("TOTAL EPISODE REWARD: ", np.array(rewards).sum())
		print("BEST EPISODE REWARD: ",np.array(rewards).max())
		print("AVERAGE EPISODE REWARD: ",np.array(rewards).mean())
		print("TIME:", time.time() - start_time)
		print("STEPS: ",steps)
		print()

	plot_reward(rewards_per_episode,range(1000))
	#TEST PHASE
	while True:
		input("PRESS ANY KEY FOR DEMONSTRATION...")
		current_state = env.reset()
		steps=0
		done = False
		rewards = []

		while not done:
			env.render()
			action = agent.act(current_state,1_000_000)
			next_state, reward, done, _ = env.step(action)
			rewards += [reward]
			current_state = next_state
			steps+=1

		print("TOTAL EPISODE REWARD: ", np.array(rewards).sum())
		print("BEST EPISODE REWARD: ",np.array(rewards).max())
		print("AVERAGE EPISODE REWARD: ",np.array(rewards).mean())
		print("TIME:", time.time() - start_time)
		print("STEPS: ",steps)
		print()

def plot_reward(rewards,episodes):
	plt.plot(episodes, rewards) 
	plt.xlabel('Episode')
	plt.ylabel('Reward') 
	plt.title('Reward evolution')
	plt.show()

if __name__ == '__main__':
	main()
