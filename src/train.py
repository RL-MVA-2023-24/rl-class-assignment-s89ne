from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
from copy import deepcopy
from evaluate import evaluate_HIV

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
class dqn_agent:
    def __init__(self, config, model, dqn_type = "simple", load_idx = None):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model
        if dqn_type == "maxmin":
            self.model1 = deepcopy(self.model).to(device)
            self.model2 = deepcopy(self.model).to(device)
        self.target_model = deepcopy(self.model).to(device)
        if dqn_type == "maxmin":
            self.target_model2 = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        if dqn_type == "maxmin":
            self.optimizer2 = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model2.parameters(), lr=lr)
        self.scheduler = config['scheduler'] if 'scheduler' in config.keys() else None
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        self.saved_models = pd.read_csv("./models/saved models.csv")
        self.model_index = len(self.saved_models)
        self.model_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        if load_idx is None:
            self.model_path = "./models/weights/model_" + str(self.model_index) + ".pt"
        else :
            self.model_index = load_idx
            self.model_path = "./models/weights/model_" + str(self.model_index) + ".pt"
            self.model.load_state_dict(torch.load(self.model_path))
        self.dqn_type = dqn_type
        self.line = {"model_index": self.model_index,
                    "model_date": self.model_date,
                    "model_path": self.model_path,
                    "episode": 0,
                    "type" : self.dqn_type,
                    "epsilon": self.epsilon_max,
                    "batch_size": self.batch_size,
                    "gamma": self.gamma,
                    "buffer_size": buffer_size,
                    "epsilon_max": self.epsilon_max,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay_period": self.epsilon_stop,
                    "epsilon_delay_decay": self.epsilon_delay,
                    "update_target_strategy": self.update_target_strategy,
                    "update_target_freq": self.update_target_freq,
                    "update_target_tau": self.update_target_tau,
                    "nb_gradient_steps": self.nb_gradient_steps,
                    "learning_rate": lr,
                    "optimizer": "Adam",
                    "scheduler": "None",
                    "criterion": "MSELoss",
                    "score": 0}
        self.saved_models_ = pd.concat([self.saved_models, pd.DataFrame(self.line, index=[0])], ignore_index=True)
        self.architecture = str(self.model)
        self.best_score = 0
        self.best_episode_cum_reward = 0

        if load_idx is None:
            print("Initializing ", self.dqn_type, " DQN...")
        else:
            print("Loading model", self.model_index, "...")

    def gradient_step_simple(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size) # sample a minibatch of transitions
            QYmax = self.target_model(Y).max(1)[0].detach() # compute the maximum target Q-value over the actions for the next state
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1)) # torch.gather is used to select the Q-values for the actions taken
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def gradient_step_double(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            max_idx = torch.argmax(self.model(Y), dim=1)
            max_idx = torch.argmax(self.target_model(Y), dim=1)
            QYmax = self.target_model(Y).gather(1, max_idx.unsqueeze(1)).detach().squeeze(1)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def gradient_step_maxmin(self):
        print("Entering maxmin, memory size : ", len(self.memory))
        if len(self.memory) > self.batch_size:
            print("Memory size : ", len(self.memory))
            X, A, R, Y, D = self.memory.sample(self.batch_size)

            QYmax = torch.minimum(self.target_model(Y), self.target_model2(Y)).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)

            QXA1 = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            QXA2 = self.model2(X).gather(1, A.to(torch.long).unsqueeze(1))

            loss = self.criterion(QXA1, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss2 = self.criterion(QXA2, update.unsqueeze(1))
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                if self.dqn_type == "double":
                    self.gradient_step_double()
                elif self.dqn_type == "simple":
                    self.gradient_step_simple()
                elif self.dqn_type == "maxmin":
                    self.gradient_step_maxmin()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                    if self.dqn_type == "maxmin":
                        self.target_model2.load_state_dict(self.model2.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
                if self.dqn_type == "maxmin":
                    target_state_dict2 = self.target_model2.state_dict()
                    model_state_dict2 = self.model2.state_dict()
                    for key in model_state_dict2:
                        target_state_dict2[key] = tau*model_state_dict2[key] + (1-tau)*target_state_dict2[key]
                    self.target_model2.load_state_dict(target_state_dict2)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:,}'.format(int(episode_cum_reward)),
                      ", l_rate ", '{:6.2e}'.format(self.optimizer.param_groups[0]['lr']),
                      sep='')
                if (int(episode_cum_reward) > self.best_episode_cum_reward) or (int(episode_cum_reward) > 1e10):
                    self.best_episode_cum_reward = int(episode_cum_reward)
                    self.line["score"] = '{:,}'.format(int(episode_cum_reward))
                    self.line["episode"] = episode
                    self.line["epsilon"] = epsilon
                    self.line["learning_rate"] = self.optimizer.param_groups[0]['lr']
                    print("Computing score...")
                    score = evaluate_HIV(self, nb_episode=1)
                    print("Eval:", '{:,}'.format(int(score)))
                    self.line["eval"] = '{:,}'.format(int(score))
                    self.saved_models_ = pd.concat([self.saved_models, pd.DataFrame(self.line, index=[0])], ignore_index=True)
                    if int(score) > self.best_score:
                        self.best_score = int(score)
                        self.save()
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)

    def save(self):
        print("Saving...")
        self.saved_models_.to_csv("./models/saved models.csv", index=False)
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))

def load_model(model_index):
      device = "cuda" if torch.cuda.is_available() else "cpu"
      df = pd.read_csv("./models/saved models.csv")
      params = df.set_index("model_index").to_dict(orient="index")[model_index]
      weights = torch.load(params["model_path"])

      env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=220)
      # Declare network
      state_dim = env.observation_space.shape[0]
      n_action = env.action_space.n

      nb_neurons=weights["2.weight"].shape[0]
      nb_layers = len(weights.keys())//2
      DQN = torch.nn.ModuleList([nn.Linear(state_dim, nb_neurons), nn.ReLU()])
      for i in range(nb_layers-2):
            DQN.append(nn.Linear(nb_neurons, nb_neurons))
            DQN.append(nn.ReLU())
      DQN.append(nn.Linear(nb_neurons, n_action))
      DQN = nn.Sequential(*DQN).to(device)
      config = {'nb_actions': env.action_space.n,
            'learning_rate': params["learning_rate"],
            'gamma': params["gamma"],
            'buffer_size': params["buffer_size"],
            'epsilon_min': params["epsilon_min"],
            'epsilon_max': params["epsilon_max"],
            'epsilon_decay_period': params["epsilon_decay_period"],
            'epsilon_delay_decay': params["epsilon_delay_decay"],
            'batch_size': params["batch_size"],
            'gradient_steps': params["nb_gradient_steps"],
            'update_target_strategy': params["update_target_strategy"],
            'update_target_freq': params["update_target_freq"],
            'update_target_tau': params["update_target_tau"],
            'criterion': torch.nn.SmoothL1Loss()}
      config['optimizer'] = torch.optim.Adam(DQN.parameters(), lr=config['learning_rate'])
      agent = dqn_agent(config, DQN, dqn_type = params["type"], load_idx = model_index)

      return agent

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        return self.agent.act(observation, use_random)

    def save(self, path):
        self.agent.save()

    def load(self):
        self.agent = load_model(23)