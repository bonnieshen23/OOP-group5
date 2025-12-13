import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

class AirHockeyEnv(gym.Env):
    # 設定渲染模式與 FPS
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        """初始化環境參數"""
        self.width = 500 
        self.height = 700
        self.render_mode = render_mode

        self.with_bot = True # 腳本機器人（用於訓練 AI 時充當對手）
        
        # 輸出x y 的施力大小，範圍 -1 到 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 接收數值: 球(x, y, vx, vy) + AI擋板(x, y) + 對手擋板(x, y)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Pygame 渲染相關變數
        self.screen = None
        self.clock = None
        self.font = None 
        self.paddle_radius = 25 # 擋板半徑
        self.ball_radius = 15   # 球半徑
        self.goal_width = 180   # 球門寬度

    def reset(self, seed=None, options=None):
        """重置環境，開始新的一局"""
        super().reset(seed=seed)
        
        # 初始化 Pymunk 物理空間
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0) # 重力
        self.space.damping = 0.999      # 空氣阻力

        # 建立四周牆壁
        self._create_walls()
        
        # 訓練模式：球隨機亂飛
        # 遊玩模式：球靜止，由玩家發球
        is_training = self.with_bot
        self.ball = self._create_ball(self.width/2, self.height/2, random_launch=is_training)
        
        # AI 在上 (y=100)、Agent在下 (y=height-100)
        self.ai_paddle = self._create_paddle(self.width/2, 100)
        self.agent_paddle = self._create_paddle(self.width/2, self.height - 100)

        # 用 mouse_body 和 PivotJoint 控制paddle
        self.mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.mouse_body.position = self.agent_paddle.body.position
        self.mouse_joint = pymunk.PivotJoint(self.mouse_body, self.agent_paddle.body, (0, 0), (0, 0))
        self.mouse_joint.max_force = 100000 # 限制最大拉力
        self.space.add(self.mouse_joint)

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        """環境步進函數：接收動作 -> 更新物理 -> 回傳狀態與獎勵"""
        
        self._apply_action(self.ai_paddle, action) # 上方擋板 
        if self.with_bot: # 下方擋板
            self._move_bot()

        # 物理引擎更新 (1/60 秒跑 10 step)
        dt = 1.0 / 60.0
        for _ in range(10):
            self.space.step(dt/10)
            self._constrain_paddle_movement() # 限制paddle不能過邊界或中線

        self.steps += 1
        
        reward = 0 # 獎勵計算
        terminated = False # 是否分出勝負
        truncated = False  # 是否超時

        ball_y = self.ball.body.position.y

        # 贏了給AI reward
        if ball_y < 0: #球進入上方球門
            reward = 10 
            terminated = True
        elif ball_y > self.height: # 球進入下方球門
            reward = -10
            terminated = True
        if ball_y < self.height / 2: # 把球壓在對方半場
            reward += 0.001

        # 防止死循環
        if self.steps > 2000:
            truncated = True
        # 如果需要渲染畫面
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _move_bot(self):
        """簡單的腳本機器人：只會左右移動追蹤球的 X 座標"""
        ball_x = self.ball.body.position.x
        current_x, current_y = self.mouse_body.position
        
        speed_limit = 8.0 # 限制 Bot 移動速度
        diff = ball_x - current_x
        
        if abs(diff) < speed_limit:
            new_x = ball_x
        else:
            new_x = current_x + speed_limit * np.sign(diff)
            
        # 限制 X 軸範圍，固定 Y 軸
        new_x = np.clip(new_x, self.paddle_radius, self.width - self.paddle_radius)
        self.mouse_body.position = (new_x, self.height - 100)

    def _create_ball(self, x, y, random_launch=False):
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, self.ball_radius)
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        
        if random_launch:
            # 隨機初速度
            import random
            rand_vx = random.uniform(-200, 200)
            rand_vy = random.uniform(-200, 200)
            body.velocity = (rand_vx, rand_vy)
            
        shape = pymunk.Circle(body, self.ball_radius)
        shape.elasticity = 1.0 # 彈性碰撞 (1.0 = 完全彈性)
        shape.friction = 0.0   # 無摩擦力
        self.space.add(body, shape)
        return shape

    def _create_paddle(self, x, y):
        # 建立擋板 (質量較大，不易被球撞飛)
        mass = 20 
        inertia = pymunk.moment_for_circle(mass, 0, self.paddle_radius)
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        shape = pymunk.Circle(body, self.paddle_radius)
        shape.elasticity = 1.0
        shape.friction = 0.0
        self.space.add(body, shape)
        return shape

    def _create_walls(self):
        # 定義靜態牆壁與球門缺口
        static_lines = [
            [(0, 0), (0, self.height)], # 左牆
            [(self.width, 0), (self.width, self.height)], # 右牆
            # 上牆 (留球門)
            [(0, 0), (self.width/2 - self.goal_width/2, 0)], 
            [(self.width/2 + self.goal_width/2, 0), (self.width, 0)], 
            # 下牆 (留球門)
            [(0, self.height), (self.width/2 - self.goal_width/2, self.height)], 
            [(self.width/2 + self.goal_width/2, self.height), (self.width, self.height)] 
        ]
        for p1, p2 in static_lines:
            shape = pymunk.Segment(self.space.static_body, p1, p2, 5)
            shape.elasticity = 1.0
            shape.friction = 0.0
            self.space.add(shape)

    def _apply_action(self, paddle, action):
        """將神經網路輸出的數值轉換為物理力"""
        force_mult = 50000 
        action = np.clip(action, -1, 1)
        # 對擋板中心施力
        paddle.body.apply_force_at_local_point((action[0] * force_mult, action[1] * force_mult))

    def _constrain_paddle_movement(self):
        """限制擋板活動範圍，防止穿模或過中線"""
        # 下方擋板限制
        p = self.agent_paddle.body.position
        new_x = np.clip(p.x, self.paddle_radius, self.width - self.paddle_radius)
        new_y = np.clip(p.y, self.height/2 + self.paddle_radius, self.height - self.paddle_radius)
        self.agent_paddle.body.position = (new_x, new_y)

        # 上方擋板限制
        p_ai = self.ai_paddle.body.position
        new_ai_x = np.clip(p_ai.x, self.paddle_radius, self.width - self.paddle_radius)
        new_ai_y = np.clip(p_ai.y, self.paddle_radius, self.height/2 - self.paddle_radius)
        self.ai_paddle.body.position = (new_ai_x, new_ai_y)

    def _get_obs(self):
        """取得當前狀態並進行歸一化 (Normalization)"""
        w, h = self.width, self.height
        bx, by = self.ball.body.position
        bvx, bvy = self.ball.body.velocity
        ax, ay = self.ai_paddle.body.position
        ox, oy = self.agent_paddle.body.position 
        # 將座標除以寬高，速度除以 1000，縮放到 0~1 或 -1~1 之間，利於神經網路訓練
        return np.array([bx/w, by/h, bvx/1000, bvy/1000, ax/w, ay/h, ox/w, oy/h], dtype=np.float32)

    def manual_move_agent(self, mouse_x, mouse_y):
        """遊玩模式用：更新滑鼠控制點的位置"""
        self.mouse_body.position = (mouse_x, mouse_y)

    def render_text(self, text, color=(0, 0, 0)):
        """在畫面上顯示文字 (如 YOU WIN)"""
        if self.screen is None: return
        if self.font is None:
            self.font = pygame.font.Font(None, 74)
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.width/2, self.height/2))
        bg_rect = text_rect.inflate(20, 20) # 文字背景框
        s = pygame.Surface((bg_rect.width, bg_rect.height))
        s.set_alpha(200) # 半透明背景
        s.fill((255, 255, 255))
        self.screen.blit(s, bg_rect.topleft)
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def render(self):
        """繪製遊戲畫面"""
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # 處理視窗關閉事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return 

        if self.screen is None: return
        self.screen.fill((255, 255, 255)) # 白底
        # 畫中線
        pygame.draw.line(self.screen, (200, 0, 0), (0, self.height//2), (self.width, self.height//2), 2)
        
        # 畫球 (紅色)
        bx, by = self.ball.body.position
        pygame.draw.circle(self.screen, (255, 0, 0), (int(bx), int(by)), self.ball_radius)
        
        # 畫 AI (藍色)
        ax, ay = self.ai_paddle.body.position
        pygame.draw.circle(self.screen, (0, 0, 255), (int(ax), int(ay)), self.paddle_radius)
        
        # 畫玩家 (藍色)
        px, py = self.agent_paddle.body.position
        pygame.draw.circle(self.screen, (0, 0, 255), (int(px), int(py)), self.paddle_radius)
        
        # 畫邊框
        pygame.draw.rect(self.screen, (0,0,0), (0,0,self.width, self.height), 5)
        
        # 畫球門線
        gw = self.goal_width
        pygame.draw.line(self.screen, (255,255,255), (self.width/2 - gw/2, 0), (self.width/2 + gw/2, 0), 5)
        pygame.draw.line(self.screen, (255,255,255), (self.width/2 - gw/2, self.height), (self.width/2 + gw/2, self.height), 5)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """關閉視窗與資源釋放"""
        if self.screen:
            pygame.quit()
            self.screen = None