import gymnasium as gym
from stable_baselines3 import PPO #用於訓練的環境
import os
import pygame
import numpy as np
from air_hockey_env import AirHockeyEnv 

# 儲存已建立模型
MODEL_PATH = "air_hockey_ppo.zip"

def train_model():
    """模式 1: 訓練 AI"""
    print("開始訓練模式...")
    # 建立環境，不需 render (畫面)
    env = AirHockeyEnv(render_mode=None)
    
    # 確保訓練時開啟 Bot，這樣 AI 才有對手可以練習
    env.with_bot = True 
    

    try: # 嘗試載入舊模型
        model = PPO.load(MODEL_PATH, env=env)
        print("載入舊模型繼續訓練...")
    except:
        # 使用 MlpPolicy (多層感知機) 策略
        model = PPO("MlpPolicy", env, verbose=1)
        print("建立新模型...")

    # 開始訓練，total_timesteps 為 AI 與環境互動的總步數
    model.learn(total_timesteps=1000000)
    model.save(MODEL_PATH)
    print(f"模型已儲存至 {MODEL_PATH}")
    env.close()

def play_game():
    """模式 2: 人機對戰"""
    print("進入遊玩模式 (人機對戰)...")
    print("按下 ESC 鍵可結束遊玩")
    
    if not os.path.exists(MODEL_PATH):
        print("找不到模型檔案，請先執行訓練模式！")
        return

    model = PPO.load(MODEL_PATH)
    
    env = AirHockeyEnv(render_mode="human")# 設定畫面 render_mode="human"
    env.with_bot = False # 遊玩模式關閉 Bot，將下半控制權交給玩家
    
    obs, _ = env.reset()
    env.render()
    running = True
    while running:
        # 視窗是否被關閉
        if env.screen is None:
            running = False
            break

        # AI 動作預測
        w, h = env.width, env.height
        bx, by = env.ball.body.position
        bvx, bvy = env.ball.body.velocity
        ai_pos = env.ai_paddle.body.position
        player_pos = env.agent_paddle.body.position
        # observation，讓 AI 覺得自己是在「標準化」的一端
        fake_obs = np.array([
            bx/w, 1 - (by/h),         # 球 Y 座標反轉
            bvx/1000, -bvy/1000,      # 球 Y 速度反轉
            ai_pos.x/w, 1 - (ai_pos.y/h),       # AI Y 座標反轉
            player_pos.x/w, 1 - (player_pos.y/h)# 玩家 Y 座標反轉
        ], dtype=np.float32)

        
        action, _ = model.predict(fake_obs)# model.predict 回傳最佳動作
        env._apply_action(env.ai_paddle, action)# 強制讓 AI 控制上方擋板 (ai_paddle)

        # 玩家動作處理
        try:
            # 獲取滑鼠位置
            mouse_x, mouse_y = pygame.mouse.get_pos()
            env.manual_move_agent(mouse_x, mouse_y)
        except pygame.error:
            running = False
            break

        # 環境步進
        # 這裡傳入 [0,0] 是因為在 play_game 中，step 函數主要負責物理更新
        # 而 AI 的動作已經在上面手動 apply 了，玩家動作則由 manual_move_agent 處理
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))

        # 遊戲結束判定
        if terminated:
            ball_y = env.ball.body.position.y
            msg = ""
            color = (0, 0, 0)
            if ball_y < 0: # 球進入上方，玩家贏
                msg = "YOU WIN!" 
                color = (0, 200, 0)
            elif ball_y > env.height: # 球進入下方，AI 贏
                msg = "AI WINS!" 
                color = (200, 0, 0)
            
            env.render_text(msg, color)
            pygame.time.wait(2000) # 顯示訊息 2 秒
            obs, _ = env.reset()

        if truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    mode = input("請選擇模式 (1: 訓練 AI, 2: 遊玩模式): ")
    if mode == "1":
        train_model()
    elif mode == "2":
        play_game()
    else:
        print("無效輸入")