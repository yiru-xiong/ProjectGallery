## Implementation of an Episodic Reinforcement, a policy-gradient learning algorithm.
# install packages to run on Colab
!apt install swig cmake libopenmpi-dev zlib1g-dev
!pip install stable-baselines[mpi]==2.10.0 box2d box2d-kengz
!pip install pyyaml h5py

import gym
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
# added pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle

# Set up enviroment to test implementation
class BanditEnv(gym.Env):
    '''
    The state is fixed (bandit setup)
    Action space: gym.spaces.Discrete(10)
    NoteL: action takes integer values
    '''
    def __init__(self):
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

    def reset(self):
        return np.array([0])

    def step(self, action):
        assert int(action) in self.action_space

        done = True
        s = np.array([0])
        r = -(action - 7)**2
        info = {}
        return s, r, done, info

# create model using PyTorch (Used in this version)
class Model_w_baseline():

    def __init__(self, env):
        self.env = env
        self.inputs_n = self.env.observation_space.shape[0]
        self.outputs_n = self.env.action_space.n
        
        # action network 
        # org: (16,16)
        self.action_net = nn.Sequential(
            nn.Linear(self.inputs_n, 128), 
            #Relu activation
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.outputs_n),
            # softmax activation
            nn.Softmax(dim=-1))
        
        # value network for baseline purpose
        self.value_net = nn.Sequential(
            nn.Linear(self.inputs_n, 32),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32, 1) 
        )
        # kernal & bias initialization
        for m in self.action_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)  
                nn.init.constant(m.bias, 0)

        for n in self.value_net:
            if isinstance(n, nn.Linear):
                nn.init.xavier_uniform_(n.weight.data)  
                nn.init.constant(n.bias, 0)

    def action_predict(self, state):
        action_probs = self.action_net(torch.FloatTensor(state))
        return action_probs
    
    def state_value(self, state):
        baseline_values = self.value_net(torch.FloatTensor(state))
        return baseline_values


class Reinforce():
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, lr, gamma, num_episodes, num_test_episodes,batch_size, model_path, model):

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.num_test_episodes = num_test_episodes
        self.env = env
        self.action_space = self.env.action_space.n
        self.model_path = model_path
        # choose Adam optimizer 
        self.model = model
    
    def discount_rewards(self, rewards):
        reward_len = len(rewards)
        vs = np.zeros_like(rewards)
        reward_tot = 0
        vs[-1] = rewards[-1]
        for r in reversed(range(0,reward_len-1)):
          reward_tot = reward_tot * self.gamma + rewards[r]
          vs[r] = reward_tot
          #normalization
          # vs = vs - np.mean(vs)
          # vs /= np.std(vs)+1e-5
        return vs
    
    def get_action(self, state):
        action_probs = self.model.action_predict(state).detach().numpy()
        # handling prob = Nan cases
        action_probs = [1e-7 if math.isnan(a) else a for a in action_probs]
        # handling probabilities do not sum to 1
        action_probs /= np.sum(action_probs)
        action = np.random.choice(self.action_space,1, p=action_probs)[0]
        return action
    
    def get_state_values(self, state):
        sv = self.model.state_value(state).detach().numpy()
        sv = [1e-7 if math.isnan(s) else s for s in sv]
        return sv

    def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        state = self.env.reset()
        done = False
        while not done:
          action = self.get_action(state)
          next_state, reward, done, info = env.step(action)
          states.append(state)
          actions.append(action)
          rewards.append(reward)
          state = next_state
        return states, actions, rewards  

    # run 100 test episodes per k training episodes 
    def test_num_episode(self):
        tot_rewards = []
        for i in range(self.num_test_episodes):
           _,_,reward = self.generate_episode()
           tot_rewards.append(np.sum(reward))
        mean_tot = np.mean(tot_rewards)
        std_reward = np.std(tot_rewards)
        return mean_tot, std_reward

    # visualize test result
    def plot_test(self, stats):
        plt.figure()
        plt.title("Reinforce Algorithm with Baseline Learning Curve")
        x_val = [s[0] for s in stats]
        y_val = [s[1] for s in stats]
        err_val = [s[2] for s in stats]
        plt.errorbar(x_val, y_val, yerr=err_val, ecolor='paleturquoise', capsize = 1)
        plt.axhline(y=200, color='gold', linestyle=':')
        plt.xlabel('episodes')
        plt.ylabel('mean cumulative reward')
        plt.show()
        plt.savefig('/content/reinforce_with_baseline_plot%s.png' %(x_val[-1]), dpi=300)

    # training + testing 
    def train(self):
        # lists to hold values to return
        stats=[]
        total_rewards = []
        plot_data = []
        action_space = self.env.action_space.n
        optimizer = optim.Adam(self.model.action_net.parameters(), lr=self.lr)
        optimizer2 = optim.Adam(self.model.value_net.parameters(), lr=self.lr)
        
        # added for batching
        batch_cnt = 0
        batch_rewards = []
        batch_actions = []
        batch_states = []

      
        for episode in range(self.num_episodes):
            states = []
            rewards = []
            actions = []
            done = False
            state = self.env.reset()
            while not done:
                # Get action based on state
                action = self.get_action(state)
                state_value = self.get_state_values(state)
                new_state, reward, done, info = self.env.step(action)
                actions.append(action)
                #action_probs.append(action_prob)
                states.append(state)
                rewards.append(reward)
                #state_values.append(state_value)
                state = new_state
      
                # If complete, batch data
                if done:
                    batch_rewards.extend(self.discount_rewards(rewards))
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    total_rewards.append(sum(rewards))
                    batch_cnt += 1
                    
                    #  update the network after reaching batch size 
                    if batch_cnt == self.batch_size:
                        optimizer.zero_grad() # initialization
                        optimizer2.zero_grad()
                        st = torch.FloatTensor(batch_states) #S: states
                        rt = torch.FloatTensor(batch_rewards) #G: discounted rewards
                        at = torch.LongTensor(batch_actions) #A: actions
                        logp = torch.log(self.model.action_predict(st))
                        state_v = self.model.state_value(st)
                        diff = rt - state_v
                        scaled_diff = diff*logp[np.arange(len(at)), at]
                        loss_p = -scaled_diff.sum()
                        #loss_p = -scaled_diff.mean()
                        loss_mse = F.mse_loss(state_v, rt)
                        loss =loss_p + loss_mse
                        # gradient update
                        loss.backward()
                        optimizer.step()
                        optimizer2.step()
                        
                        batch_rewards = []
                        batch_actions = []
                        batch_states = []
                        batch_cnt = 0
                        
                    # training statistics
                    stats.append([episode+1, np.mean(total_rewards[-100:]),np.std(total_rewards[-100:])])
                    print("\rEpisode: {}, Average of last 100: {:.2f}".format(
                    episode + 1, np.mean(total_rewards[-100:])), end="")                   
             
            if (episode+1)%100 == 0:
                #torch.save(self.model, self.model_path+str("model%s.pt"%(episode+1)))
                #testing statisticcs
                avg_reward, std_reward=self.test_num_episode()
                print("\rEpisode: {} Average : {:.2f} Std:{:.2f}".format(
                episode + 1, avg_reward, std_reward, end=""))
                plot_data.append([episode+1,avg_reward,std_reward]) 

            if (episode+1) % 5000 == 0:
                self.plot_test(plot_data)
                with open("/content/plot_baseline_data%s.txt"%(episode+1),"wb") as fp:
                     pickle.dump(plot_data, fp)


        self.plot_test(plot_data)
        #torch.save(self.model, "/content/drive/My Drive/reinforce/model%s.pt"%(episode+1))
        return total_rewards, stats, plot_data

## training 
# !rm -rf /content/models_baseline/
# !mkdir /content/models_baseline/

seed = 3
#np.random.seed(seed)
#env = BanditEnv()
env = gym.make('LunarLander-v2')
env.seed(seed)
torch.manual_seed(seed)
model = Model_w_baseline(env)
lr = 1e-3
gamma = 0.99 #0 for bandit #0.99 for LunarLander
num_episodes = 25000
num_test_episodes = 100
batch_size = 32
model_path = "/content/"
agent = Reinforce(env, lr, gamma, num_episodes, num_test_episodes,batch_size, model_path,model)
total_rewards, stats, plot_data = agent.train()

# save outputs from training and testing 
# with open("/content/baseline_rewards.txt","wb") as fp:
#      pickle.dump(total_rewards, fp)

# with open("/content/train_baseline_stats.txt","wb") as fp:
#      pickle.dump(stats, fp)
# with open("/content/test_baseline_stats.txt","wb") as fp:
#      pickle.dump(plot_data, fp)