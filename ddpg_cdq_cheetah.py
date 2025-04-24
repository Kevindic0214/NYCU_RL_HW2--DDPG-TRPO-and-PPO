# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG with Clipped Double Q (CDQ) for HalfCheetah

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_cdq/HalfCheetah-v2")

def soft_update(target, source, tau):
    """
    軟更新目標網絡參數
    target = (1 - tau) * target + tau * source
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    硬更新目標網絡參數
    target = source
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    經驗回放緩衝區
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """
    Ornstein-Uhlenbeck 噪聲生成器，用於探索
    """
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    """
    策略網絡 (Actor) - 決定在給定狀態下的最佳動作
    """
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # 儲存動作空間的縮放和偏移量，用於將輸出轉換到合適的範圍
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)
        
        # 構建網絡架構
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()  # 輸出範圍限制在[-1, 1]
        )
        
    def forward(self, inputs):
        """前向傳播"""
        return self.net(inputs)

class Critic(nn.Module):
    """
    評論家網絡 (Critic) - 評估狀態-動作對的價值
    這是CDQ中的單個Q網絡
    """
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # 構建網絡架構
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs, actions):
        """前向傳播，結合狀態和動作作為輸入"""
        x = torch.cat([inputs, actions], dim=1)
        return self.net(x)

class DDPG_CDQ(object):
    """
    帶有截斷雙Q (Clipped Double Q) 技術的DDPG實現
    """
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, 
                 lr_a=1e-4, lr_c=1e-3, noise_sigma=0.2, noise_clip=0.5):
        self.num_inputs = num_inputs
        self.action_space = action_space
        
        # 策略網絡和目標策略網絡
        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        # 雙Q網絡實現：我們需要兩個評論家網絡和它們各自的目標網絡
        self.critic1 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic1_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=lr_c)
        
        self.critic2 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic2_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau
        
        # 目標策略平滑的參數
        self.noise_sigma = noise_sigma  # 目標動作噪聲的標準差
        self.noise_clip = noise_clip    # 目標動作噪聲的最大限制

        # 硬更新：將目標網絡的參數設置為與主網絡相同
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

    def select_action(self, state, action_noise=None):
        """選擇動作，可選添加探索噪聲"""
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        # 添加噪聲以進行探索
        if action_noise is not None:
            noise = action_noise.noise()
            mu += torch.FloatTensor(noise).float()
            
        # 將動作裁剪到合法範圍
        mu = torch.clamp(mu, min=torch.FloatTensor(self.action_space.low), max=torch.FloatTensor(self.action_space.high))
        return mu

    def update_parameters(self, batch):
        """更新網絡參數"""
        # 將批次格式轉換為張量
        if isinstance(batch, list):
            batch = Transition(*zip(*batch))
            
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        # ----- 實現CDQ算法的關鍵部分 -----
        with torch.no_grad():
            # 生成下一個動作（目標策略輸出）
            next_actions = self.actor_target(next_state_batch)
            
            # 添加並裁剪噪聲，實現TD3中的目標策略平滑
            noise = torch.randn_like(next_actions) * self.noise_sigma
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            smoothed_next_actions = next_actions + noise
            
            # 確保動作在合法範圍內
            smoothed_next_actions = torch.clamp(
                smoothed_next_actions, 
                torch.FloatTensor(self.action_space.low), 
                torch.FloatTensor(self.action_space.high)
            )
            
            # 雙Q網絡評估下一個狀態-動作對的價值
            q1_next = self.critic1_target(next_state_batch, smoothed_next_actions)
            q2_next = self.critic2_target(next_state_batch, smoothed_next_actions)
            
            # 取兩個Q值的最小值（截斷技術核心）
            q_next = torch.min(q1_next, q2_next)
            
            # 計算目標Q值
            q_target = reward_batch.unsqueeze(1) + (self.gamma * q_next * mask_batch.unsqueeze(1))

        # 更新第一個評論家網絡
        q1_current = self.critic1(state_batch, action_batch)
        value_loss1 = F.smooth_l1_loss(q1_current, q_target)
        self.critic1_optim.zero_grad()
        value_loss1.backward()
        self.critic1_optim.step()
        
        # 更新第二個評論家網絡
        q2_current = self.critic2(state_batch, action_batch)
        value_loss2 = F.smooth_l1_loss(q2_current, q_target)
        self.critic2_optim.zero_grad()
        value_loss2.backward()
        self.critic2_optim.step()

        # 策略網絡的延遲更新
        # 我們只使用第一個評論家網絡來更新策略
        policy_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # 軟更新目標網絡
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

        return (value_loss1.item() + value_loss2.item())/2, policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic1_path=None, critic2_path=None):
        """保存模型參數"""
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_cdq_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic1_path is None:
            critic1_path = "preTrained/ddpg_cdq_critic1_{}_{}_{}".format(env_name, timestamp, suffix)
        if critic2_path is None:
            critic2_path = "preTrained/ddpg_cdq_critic2_{}_{}_{}".format(env_name, timestamp, suffix)
            
        print('Saving models to {}, {}, and {}'.format(actor_path, critic1_path, critic2_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load_model(self, actor_path, critic1_path, critic2_path):
        """載入模型參數"""
        print('Loading models from {}, {}, and {}'.format(actor_path, critic1_path, critic2_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic1_path is not None: 
            self.critic1.load_state_dict(torch.load(critic1_path))
        if critic2_path is not None:
            self.critic2.load_state_dict(torch.load(critic2_path))

def train(env, env_name):
    """訓練主函數"""
    num_episodes = 500  # 總共訓練的回合數
    gamma = 0.995       # 折扣因子
    tau = 0.02          # 軟更新係數
    hidden_size = 128   # 隱藏層大小
    noise_scale = 0.1   # OU噪聲的比例
    replay_size = 100000  # 回放緩衝區大小
    batch_size = 256    # 批次大小
    updates_per_step = 1  # 每步更新次數
    print_freq = 1      # 打印頻率
    
    # TD3特有的超參數
    noise_sigma = 0.2   # 目標策略平滑的噪聲標準差
    noise_clip = 0.5    # 目標策略平滑的噪聲裁剪值
    
    ewma_reward = 0     # 指數加權移動平均獎勵
    rewards = []        # 獎勵歷史
    ewma_reward_history = []  # EWMA獎勵歷史
    total_numsteps = 0  # 總步數
    updates = 0         # 更新次數

    # 初始化代理、噪聲生成器和回放緩衝區
    agent = DDPG_CDQ(env.observation_space.shape[0], env.action_space, 
                     gamma, tau, hidden_size, noise_sigma=noise_sigma, noise_clip=noise_clip)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    # 訓練迴圈
    for i_episode in range(num_episodes):
        # 重置噪聲和環境
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state = torch.from_numpy(env.reset()).float().unsqueeze(0)

        episode_reward = 0
        while True:
            # 選擇動作並與環境互動
            action = agent.select_action(state, ounoise)
            next_obs, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            episode_reward += reward
            
            # 將經驗存入回放緩衝區
            mask = torch.tensor([0.0 if done else 1.0], dtype=torch.float32)
            reward_t = torch.tensor([reward], dtype=torch.float32)
            memory.push(state, action, mask, next_state, reward_t)

            state = next_state
            total_numsteps += 1            
            
            # 更新網絡
            if len(memory) >= batch_size:
                for _ in range(updates_per_step):
                    transitions = memory.sample(batch_size)
                    value_loss, policy_loss = agent.update_parameters(transitions)
                    updates += 1
                    # 紀錄損失值
                    writer.add_scalar('Loss/Value', value_loss, updates)
                    writer.add_scalar('Loss/Policy', policy_loss, updates)
            
            if done:
                break

        rewards.append(episode_reward)
        
        # 評估當前策略
        if i_episode % print_freq == 0:
            state = torch.tensor(env.reset()).float().unsqueeze(0)
            episode_reward = 0
            t = 0
            while True:
                # 無噪聲的評估
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                episode_reward += reward
                next_state = torch.tensor(next_state).float().unsqueeze(0)
                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # 更新EWMA獎勵並記錄結果
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}, critic loss: {:.2f}, actor loss: {:.2f}".format(
                i_episode, t, rewards[-1], ewma_reward, value_loss, policy_loss))
            
            # Tensorboard記錄
            writer.add_scalar('Reward/Episode', episode_reward, i_episode)
            writer.add_scalar('Reward/EWMA', ewma_reward, i_episode)
            
    # 保存訓練好的模型
    agent.save_model(env_name, '.pth')
 
if __name__ == '__main__':
    # 為了可重現結果，固定隨機種子
    random_seed = 10  
    env = gym.make('HalfCheetah-v2')
    env.reset(seed=random_seed)  
    torch.manual_seed(random_seed)  
    train(env, env_name='HalfCheetah-v2')