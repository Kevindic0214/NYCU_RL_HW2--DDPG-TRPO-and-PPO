import gym
import torch
import numpy as np
import time
import torch.nn as nn

# 定義Actor網路模型，需與訓練時的結構一致
class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # 與訓練時的Actor網路結構保持一致
        self.action_scale = torch.FloatTensor(action_space.high - action_space.low) / 2.0
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )       
        
    def forward(self, inputs):
        x = self.net(inputs)
        return x * self.action_scale + self.action_bias

# 定義一個策略函數，使用Actor網路選擇動作
def policy(observation, actor):
    # 將觀察轉換為PyTorch張量
    state = torch.FloatTensor(observation).unsqueeze(0)
    
    # 使用Actor網路預測動作
    with torch.no_grad():
        action = actor(state).detach().numpy()[0]
    
    return action

# 定義測試函數
def test_model(actor_path, episodes=5, max_steps=200, delay=0.01):
    """
    使用預先訓練好的Actor模型在環境中執行並渲染結果
    
    參數:
        actor_path: 預訓練模型的路徑
        episodes: 要執行的回合數
        max_steps: 每個回合的最大步數
        delay: 每步之間的延遲時間（秒）
    """
    # 創建環境 - 不使用render_mode參數
    env = gym.make('Pendulum-v1')
    
    # 獲取環境的狀態和動作空間維度
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    
    # 初始化Actor模型
    hidden_size = 128  # 使用與訓練時相同的隱藏層大小
    actor = Actor(hidden_size, state_dim, action_space)
    
    # 載入預訓練的模型權重
    print(f"載入模型權重從: {actor_path}")
    try:
        actor.load_state_dict(torch.load(actor_path))
        actor.eval()  # 設置為評估模式
        print("模型載入成功！")
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return
    
    # 執行測試
    total_rewards = []
    
    for episode in range(episodes):
        # 按照官方示例重置環境獲取初始狀態，使用return_info=True
        try:
            observation, info = env.reset(seed=42+episode, return_info=True)
        except Exception as e:
            print(f"使用新API重置環境時出錯: {e}")
            print("嘗試使用傳統方式重置環境")
            observation = env.reset()
            info = {}
        
        print(f"\n開始測試第 {episode+1}/{episodes} 回合")
        
        episode_reward = 0
        
        for step in range(max_steps):
            # 按照官方示例進行渲染
            env.render()
            
            # 使用策略選擇動作
            action = policy(observation, actor)
            
            # 執行動作
            observation, reward, done, info = env.step(action)
            
            # 更新累計獎勵
            episode_reward += reward
            
            # 每50步或最後一步輸出一次狀態
            if step % 50 == 0 or step == max_steps - 1:
                print(f"  步驟 {step+1}: 獎勵 = {reward:.3f}, 累計獎勵 = {episode_reward:.3f}")
            
            # 延遲以便觀察
            time.sleep(delay)
            
            # 如果回合結束則跳出迴圈
            if done:
                print(f"  回合在步驟 {step+1} 結束")
                break
        
        total_rewards.append(episode_reward)
        print(f"第 {episode+1} 回合結束，總步數: {step+1}，總獎勵: {episode_reward:.3f}")
    
    # 輸出總體統計信息
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n測試完成！共 {episodes} 個回合，平均獎勵: {avg_reward:.3f}")
    print(f"最高獎勵: {max(total_rewards):.3f}, 最低獎勵: {min(total_rewards):.3f}")
    
    # 關閉環境
    env.close()

if __name__ == "__main__":
    # 硬編碼的參數設定
    model_path = "preTrained/ddpg_actor_Pendulum-v1_04232025_164317_.pth"  # 模型檔案路徑
    num_episodes = 5               # 測試回合數
    max_steps = 200               # 每個回合最大步數
    step_delay = 0.01             # 每步延遲時間
    
    # 執行測試
    test_model(
        model_path,
        episodes=num_episodes,
        max_steps=max_steps,
        delay=step_delay
    )