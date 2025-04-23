import gym
import torch
import time
import numpy as np
from ddpg import Actor

def test_model(actor_path, episodes=5, max_steps=200, delay=0.01, random_seed=42):
    """
    測試預訓練DDPG模型在Pendulum-v1環境的表現
    
    參數:
        actor_path: 預訓練模型的路徑
        episodes: 要執行的回合數
        max_steps: 每個回合的最大步數
        delay: 每步之間的延遲時間（秒）
        random_seed: 隨機種子，用於確保實驗的可重複性
    """
    # 設定隨機種子以確保可重複性
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 創建環境
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    
    # 初始化並載入模型
    actor = Actor(128, state_dim, env.action_space)
    try:
        actor.load_state_dict(torch.load(actor_path))
        actor.eval()
        print(f"成功載入模型：{actor_path}")
    except Exception as e:
        print(f"載入模型失敗：{e}")
        return
    
    # 執行測試回合
    rewards = []
    
    for episode in range(episodes):
        # 為每個回合設定不同但確定的種子
        episode_seed = random_seed + episode
        env.seed(episode_seed)  # Gym 0.23.1 中設定環境種子的方法
        observation = env.reset()
        
        print(f"\n執行第 {episode+1}/{episodes} 回合 (種子: {episode_seed})")
        episode_reward = 0
        
        for step in range(max_steps):
            # 渲染環境
            env.render()
            
            # 預測動作
            state = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                action = actor(state).detach().numpy()[0]
            
            # 執行動作並獲取結果
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 顯示進度（僅在關鍵時刻）
            if step % 50 == 0 or step == max_steps - 1:
                print(f"  步驟 {step+1}: 獎勵 = {reward:.3f}, 累計 = {episode_reward:.3f}")
            
            time.sleep(delay)
            if done:
                break
        
        rewards.append(episode_reward)
        print(f"回合 {episode+1} 完成，總獎勵: {episode_reward:.3f}")
    
    # 計算統計資訊
    avg_reward = sum(rewards) / len(rewards)
    print(f"\n測試結果：{episodes} 回合平均獎勵: {avg_reward:.3f}")
    print(f"最高: {max(rewards):.3f}, 最低: {min(rewards):.3f}")
    
    env.close()

if __name__ == "__main__":
    test_model(
        actor_path="preTrained/ddpg_actor_Pendulum-v1_04232025_164317_.pth",
        random_seed=42  # 設定隨機種子
    )