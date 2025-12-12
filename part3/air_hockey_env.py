import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.width = 500
        self.height = 700 
        self.render_mode = render_mode
        
        # === 新增：是否開啟自動陪練對手 ===
        # 預設為 True (訓練時用)，遊玩時我們要把它關掉
        self.with_bot = True 
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.font = None 
        
        self.paddle_radius = 25
        self.ball_radius = 15
        self.goal_width = 180

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.999 

        self._create_walls()
        
        # === 修改點：依據模式決定球是否要亂跑 ===
        # 如果是訓練模式 (with_bot 為 True)，則 random_launch 開啟 (球會有隨機初速度)
        # 如果是遊玩模式 (with_bot 為 False)，則 random_launch 關閉 (球是靜止的)
        is_training = self.with_bot
        self.ball = self._create_ball(self.width/2, self.height/2, random_launch=is_training)
        # =====================================
        
        self.ai_paddle = self._create_paddle(self.width/2, 100)
        self.agent_paddle = self._create_paddle(self.width/2, self.height - 100)

        # ... (以下滑鼠關節設定保持不變) ...
        self.mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.mouse_body.position = self.agent_paddle.body.position
        self.mouse_joint = pymunk.PivotJoint(self.mouse_body, self.agent_paddle.body, (0, 0), (0, 0))
        self.mouse_joint.max_force = 100000 
        self.space.add(self.mouse_joint)

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. AI (上方) 動作
        self._apply_action(self.ai_paddle, action)
        
        # 2. 對手 (下方) 動作：如果是訓練模式，讓 Bot 自動跑
        if self.with_bot:
            self._move_bot()

        dt = 1.0 / 60.0
        for _ in range(10):
            self.space.step(dt/10)
            self._constrain_paddle_movement()

        self.steps += 1
        
        reward = 0
        terminated = False
        truncated = False

        ball_y = self.ball.body.position.y

        # 進球判定與獎勵 (為了讓訓練更有效，這裡可以稍微加重獎勵)
        if ball_y < 0: 
            reward = 10 
            terminated = True
        elif ball_y > self.height: 
            reward = -10
            terminated = True
        
        # 簡單的獎勵引導：如果球在上方半場(壓制對手)，給一點點獎勵
        if ball_y < self.height / 2:
            reward += 0.001

        if self.steps > 2000:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    # === 新增：簡單的追球機器人 ===
    def _move_bot(self):
        # 取得球的 X 座標
        ball_x = self.ball.body.position.x
        current_x, current_y = self.mouse_body.position
        
        # 簡單邏輯：Bot 想要移動到與球相同的 X 軸
        # 設定一個移動速度限制，不然它會無敵 (例如每幀最多移動 8 像素)
        speed_limit = 8.0
        
        diff = ball_x - current_x
        
        if abs(diff) < speed_limit:
            new_x = ball_x
        else:
            # 往球的方向移動 speed_limit 的距離
            new_x = current_x + speed_limit * np.sign(diff)
            
        # 限制不要跑出牆壁
        new_x = np.clip(new_x, self.paddle_radius, self.width - self.paddle_radius)
        
        # 更新隱形滑鼠位置 (Y 軸保持在防守線上)
        self.mouse_body.position = (new_x, self.height - 100)

    # ... (以下 _create_ball, _create_paddle 等函式保持不變) ...
    def _create_ball(self, x, y, random_launch=False):
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, self.ball_radius)
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        
        if random_launch:
            # 隨機產生 X 和 Y 方向的初速度
            # 範圍可以根據手感調整，這裡設為 -200 到 200
            import random
            rand_vx = random.uniform(-200, 200)
            rand_vy = random.uniform(-200, 200)
            body.velocity = (rand_vx, rand_vy)
            
        shape = pymunk.Circle(body, self.ball_radius)
        shape.elasticity = 1.0 
        shape.friction = 0.0
        self.space.add(body, shape)
        return shape

    def _create_paddle(self, x, y):
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
        static_lines = [
            [(0, 0), (0, self.height)], 
            [(self.width, 0), (self.width, self.height)], 
            [(0, 0), (self.width/2 - self.goal_width/2, 0)], 
            [(self.width/2 + self.goal_width/2, 0), (self.width, 0)], 
            [(0, self.height), (self.width/2 - self.goal_width/2, self.height)], 
            [(self.width/2 + self.goal_width/2, self.height), (self.width, self.height)] 
        ]
        for p1, p2 in static_lines:
            shape = pymunk.Segment(self.space.static_body, p1, p2, 5)
            shape.elasticity = 1.0
            shape.friction = 0.0
            self.space.add(shape)

    def _apply_action(self, paddle, action):
        force_mult = 50000 
        action = np.clip(action, -1, 1)
        paddle.body.apply_force_at_local_point((action[0] * force_mult, action[1] * force_mult))

    def _constrain_paddle_movement(self):
        p = self.agent_paddle.body.position
        new_x = np.clip(p.x, self.paddle_radius, self.width - self.paddle_radius)
        new_y = np.clip(p.y, self.height/2 + self.paddle_radius, self.height - self.paddle_radius)
        self.agent_paddle.body.position = (new_x, new_y)

        p_ai = self.ai_paddle.body.position
        new_ai_x = np.clip(p_ai.x, self.paddle_radius, self.width - self.paddle_radius)
        new_ai_y = np.clip(p_ai.y, self.paddle_radius, self.height/2 - self.paddle_radius)
        self.ai_paddle.body.position = (new_ai_x, new_ai_y)

    def _get_obs(self):
        w, h = self.width, self.height
        bx, by = self.ball.body.position
        bvx, bvy = self.ball.body.velocity
        ax, ay = self.ai_paddle.body.position
        ox, oy = self.agent_paddle.body.position 
        return np.array([bx/w, by/h, bvx/1000, bvy/1000, ax/w, ay/h, ox/w, oy/h], dtype=np.float32)

    def manual_move_agent(self, mouse_x, mouse_y):
        self.mouse_body.position = (mouse_x, mouse_y)

    def render_text(self, text, color=(0, 0, 0)):
        if self.screen is None: return
        if self.font is None:
            self.font = pygame.font.Font(None, 74)
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.width/2, self.height/2))
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface((bg_rect.width, bg_rect.height))
        s.set_alpha(200)
        s.fill((255, 255, 255))
        self.screen.blit(s, bg_rect.topleft)
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return 

        if self.screen is None: return
        self.screen.fill((255, 255, 255)) 
        pygame.draw.line(self.screen, (200, 0, 0), (0, self.height//2), (self.width, self.height//2), 2)
        bx, by = self.ball.body.position
        pygame.draw.circle(self.screen, (255, 0, 0), (int(bx), int(by)), self.ball_radius)
        ax, ay = self.ai_paddle.body.position
        pygame.draw.circle(self.screen, (0, 0, 255), (int(ax), int(ay)), self.paddle_radius)
        px, py = self.agent_paddle.body.position
        pygame.draw.circle(self.screen, (0, 0, 255), (int(px), int(py)), self.paddle_radius)
        pygame.draw.rect(self.screen, (0,0,0), (0,0,self.width, self.height), 5)
        gw = self.goal_width
        pygame.draw.line(self.screen, (255,255,255), (self.width/2 - gw/2, 0), (self.width/2 + gw/2, 0), 5)
        pygame.draw.line(self.screen, (255,255,255), (self.width/2 - gw/2, self.height), (self.width/2 + gw/2, self.height), 5)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None