import gym
import torch
import time
import numpy as np
from ddpg_cheetah import Actor

def test_model(actor_path, episodes=5, max_steps=1000, delay=0.01, seed=42):
    """測試DDPG模型在HalfCheetah-v2環境的表現"""
    # 設置隨機種子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 創建環境
    env = gym.make('HalfCheetah-v2')
    env.seed(seed)
    
    # 載入模型
    state_dim = env.observation_space.shape[0]
    actor = Actor(128, state_dim, env.action_space)
    try:
        actor.load_state_dict(torch.load(actor_path))
        actor.eval()  # 設置為評估模式
        print(f"成功載入模型: {actor_path}")
        print(f"狀態維度: {state_dim}, 動作維度: {env.action_space.shape[0]}")
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return
    
    # 執行測試
    rewards = []
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            env.render()
            
            # 選擇動作
            state = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = actor(state).detach().numpy()[0]
            
            # 執行動作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 只在關鍵時刻顯示資訊
            if step % 200 == 0:
                velocity = obs[8]  # 通常是前進速度
                print(f"步驟 {step}: 速度={velocity:.2f}, 獎勵={reward:.2f}")
            
            time.sleep(delay)
            if done: break
            
        rewards.append(episode_reward)
        print(f"回合 {episode+1} 完成: 總獎勵={episode_reward:.2f}")
    
    print(f"\n測試結果: 平均獎勵={sum(rewards)/len(rewards):.2f}")
    env.close()

if __name__ == "__main__":
    test_model("preTrained/ddpg_actor_HalfCheetah-v2_04232025_173611_.pth")