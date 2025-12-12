import gymnasium as gym
from stable_baselines3 import PPO
import os
import pygame
import numpy as np
from air_hockey_env import AirHockeyEnv 

MODEL_PATH = "air_hockey_ppo.zip"

def train_model():
    print("開始訓練模式...")
    # 建立環境
    env = AirHockeyEnv(render_mode=None)
    
    # 確保訓練時 Bot 是開啟的
    env.with_bot = True 
    
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print("載入舊模型繼續訓練...")
    except:
        model = PPO("MlpPolicy", env, verbose=1)
        print("建立新模型...")

    # 你可以增加步數，因為現在對手會動了，訓練需要久一點才能贏
    model.learn(total_timesteps=1000000)
    model.save(MODEL_PATH)
    print(f"模型已儲存至 {MODEL_PATH}")
    env.close()

def play_game():
    print("進入遊玩模式 (人機對戰)...")
    print("按下 ESC 鍵可結束遊玩")
    
    if not os.path.exists(MODEL_PATH):
        print("找不到模型檔案，請先執行訓練模式！")
        return

    model = PPO.load(MODEL_PATH)
    env = AirHockeyEnv(render_mode="human")
    
    # 【關鍵】遊玩模式關閉 Bot，讓滑鼠完全控制
    env.with_bot = False 
    
    obs, _ = env.reset()
    env.render()
    
    running = True
    while running:
        if env.screen is None:
            running = False
            break

        # 1. AI 動作
        w, h = env.width, env.height
        bx, by = env.ball.body.position
        bvx, bvy = env.ball.body.velocity
        ai_pos = env.ai_paddle.body.position
        player_pos = env.agent_paddle.body.position

        fake_obs = np.array([
            bx/w, 1 - (by/h),
            bvx/1000, -bvy/1000,
            ai_pos.x/w, 1 - (ai_pos.y/h),
            player_pos.x/w, 1 - (player_pos.y/h)
        ], dtype=np.float32)

        action, _ = model.predict(fake_obs)
        env._apply_action(env.ai_paddle, action)

        # 2. 玩家動作
        try:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            env.manual_move_agent(mouse_x, mouse_y)
        except pygame.error:
            running = False
            break

        # 3. 步進
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))

        if terminated:
            ball_y = env.ball.body.position.y
            msg = ""
            color = (0, 0, 0)
            if ball_y < 0: 
                msg = "YOU WIN!"
                color = (0, 200, 0)
            elif ball_y > env.height: 
                msg = "AI WINS!"
                color = (200, 0, 0)
            
            env.render_text(msg, color)
            pygame.time.wait(2000)
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