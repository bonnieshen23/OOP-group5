import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

class AirHockeyEnv(gym.Env):
    # 設定渲染模式與 FPS 
        # human : 人類觀看的模式，顯示在視窗上
        # rgb_array : 訓練用，回傳np.array，而不顯示畫面
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
        super().reset(seed=seed)    # 處理隨機種子與環境基底行為
        
        # 建立新的 Pymunk 物理空間，並初始化
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
        self.mouse_body.position = self.agent_paddle.body.position  # 玩家滑鼠的控制點
        self.mouse_joint = pymunk.PivotJoint(self.mouse_body, self.agent_paddle.body, (0, 0), (0, 0))
        self.mouse_joint.max_force = 100000 # 限制最大拉力
        self.space.add(self.mouse_joint)    # 將joint加入物理空間

        self.steps = 0
        # 回傳 (observation, info)
        return self._get_obs(), {}

    def step(self, action):
        """環境步進函數：接收動作 -> 更新物理 -> 回傳狀態與獎勵"""
        
        self._apply_action(self.ai_paddle, action) # 上方擋板 
        if self.with_bot: # 若是訓練模式(有bot)，啟動簡單腳本機器人
            self._move_bot()

        # 物理引擎更新 (1/60 秒跑 10 step)
        dt = 1.0 / 60.0
        for _ in range(10):
            self.space.step(dt/10)  # 在pymunk建立的空間進行10次子步模擬
            self._constrain_paddle_movement() # 限制paddle不能過邊界或中線

        # 更新子步計數
        self.steps += 1
        
        reward = 0 # 獎勵計算
        terminated = False # 是否分出勝負
        truncated = False  # 是否超時

        ball_y = self.ball.body.position.y

        # 贏了給AI reward
        if ball_y < 0: #球進入上方球門
            reward = -10 
            terminated = True
        elif ball_y > self.height: # 球進入下方球門
            reward = 10
            terminated = True
        if ball_y > self.height / 2: # 把球壓在對方半場
            reward += 0.001

        # 防止死循環
        if self.steps > 2000:
            truncated = True
        # 如果需要渲染畫面
        if self.render_mode == "human":
            self.render()

        # 回傳 (obs, reward, terminated, truncated, info)
        return self._get_obs(), reward, terminated, truncated, {}

    def _move_bot(self):
        """簡單的腳本機器人：只會左右移動追蹤球的 X 座標"""
        ball_x = self.ball.body.position.x  # 抓出球的 X 座標
        op_x, op_y = self.mouse_body.position # 抓出目前擋板的 X 座標
        
        speed_limit = 8.0 # 限制 Bot 移動速度
        diff = ball_x - op_x
        
        # 判斷腳本機器人在速度限制下，能不能趕到球的X座標位置
        if abs(diff) < speed_limit:
            new_x = ball_x
        else:
            new_x = op_x + speed_limit * np.sign(diff)
            
        # 限制 X 軸範圍，固定 Y 軸
        # 若new_x超出邊界，就把它夾回邊界內(new_x : self.paddle_radius ~ self.width - self.paddle_radius)
        new_x = np.clip(new_x, self.paddle_radius, self.width - self.paddle_radius)
        # mose_body在這裡被用來控制腳本機器人的擋板位置(固定y座標)
        self.mouse_body.position = (new_x, self.height - 100)   

    def _create_ball(self, x, y, random_launch=False):
        mass = 1    # 眾量設為1
        # 計算轉動慣量(mass, 內半徑, 外半徑)，當碰撞時計算角動量
        inertia = pymunk.moment_for_circle(mass, 0, self.ball_radius)
        # 建立動態剛體
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        
        # 當訓練模式時，會隨機發球(給予初速度)
        if random_launch:
            # 隨機初速度
            import random
            rand_vx = random.uniform(-200, 200)
            rand_vy = random.uniform(-200, 200)
            body.velocity = (rand_vx, rand_vy)
            
        shape = pymunk.Circle(body, self.ball_radius)   # 設定為圓形的碰撞體
        shape.elasticity = 1.0 # 彈性碰撞 (完全反彈)
        shape.friction = 0.0   # 無摩擦力
        self.space.add(body, shape) # 將球加入space中
        return shape

    def _create_paddle(self, x, y):
        # 建立擋板 (質量較大，不易被球撞飛)
        mass = 20 
        # 計算轉動慣量(mass, 內半徑, 外半徑)，當碰撞時計算角動量
        inertia = pymunk.moment_for_circle(mass, 0, self.paddle_radius)
        # 建立動態剛體
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        shape = pymunk.Circle(body, self.paddle_radius)
        shape.elasticity = 1.0  # 彈性碰撞(完全反彈)
        shape.friction = 0.0    # 無摩擦力
        shape.filter = pymunk.ShapeFilter(group=1)  # 無視檔板之間的碰撞
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
        """將神經網路輸出的數值(推力方向與比例)轉換為物理力"""
        force_mult = 50000  # 力的縮放係數
        action = np.clip(action, -1, 1) # 限制動作範圍在 -1 到 1，避免過大的力
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
        """取得當前狀態並進行Normalization"""
        w, h = self.width, self.height
        bx, by = self.ball.body.position
        bvx, bvy = self.ball.body.velocity
        ax, ay = self.ai_paddle.body.position
        ox, oy = self.agent_paddle.body.position 
        # 將座標除以寬高，速度除以 1000，縮放到 0~1 或 -1~1 之間，神經網路比較好學
        # [球座標x/w, 球座標y/h, 球速度x/1000, 球速度y/1000, AI擋板x/w, AI擋板y/h, 玩家擋板x/w, 玩家擋板y/h]
        return np.array([bx/w, by/h, bvx/1000, bvy/1000, ax/w, ay/h, ox/w, oy/h], dtype=np.float32)

    def manual_move_agent(self, mouse_x, mouse_y):
        """遊玩模式用：更新滑鼠控制點的位置"""
        # 將真實滑鼠游標位置賦予給"代表滑鼠的物理剛體"
        self.mouse_body.position = (mouse_x, mouse_y)

    def render_text(self, text, color=(0, 0, 0)):
        """在畫面上顯示文字 (如 YOU WIN)"""
        # 若視窗關閉就不執行
        if self.screen is None: return
        # 初始化字型(只執行一次)
        if self.font is None:
            self.font = pygame.font.Font(None, 74)
        # 將文字轉成畫面 -> (文字, 抗鋸齒, 顏色)
        text_surface = self.font.render(text, True, color)
        # 取得文字的外框，並固定在畫面中間
        text_rect = text_surface.get_rect(center=(self.width/2, self.height/2))
        bg_rect = text_rect.inflate(20, 20) # 文字背景框
        s = pygame.Surface((bg_rect.width, bg_rect.height)) # 建立文字背後的色塊
        s.set_alpha(200) # 半透明背景
        s.fill((255, 255, 255)) # 白色的
        self.screen.blit(s, bg_rect.topleft)    # s貼在bg_rect的左上角
        self.screen.blit(text_surface, text_rect)   # 文字貼在tec_rect的位置
        pygame.display.flip()   # 使畫面更新

    def render(self):
        """繪製遊戲畫面"""
        # 建立視窗(只執行一次)
        if self.screen is None:
            # 啟動視窗系統、輸入裝置、字形系統
            pygame.init()
            pygame.font.init()
            # 在視窗上建立畫布
            self.screen = pygame.display.set_mode((self.width, self.height))
            # 設定畫面更新速度
            self.clock = pygame.time.Clock()

        # 處理視窗關閉的操作(點X或按下Esc便結束)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return 

        # 若close()了，就停止繪製
        if self.screen is None: return
        self.screen.fill((255, 255, 255)) # 畫白底(避免殘影)
        # 畫中線 -> (位置, 顏色, 起點, 終點, 寬度)
        pygame.draw.line(self.screen, (200, 0, 0), (0, self.height//2), (self.width, self.height//2), 2)
        
        # 畫球(紅色)
        bx, by = self.ball.body.position
        pygame.draw.circle(self.screen, (255, 0, 0), (int(bx), int(by)), self.ball_radius)
        
        # 畫AI(藍色)
        ax, ay = self.ai_paddle.body.position
        pygame.draw.circle(self.screen, (0, 0, 255), (int(ax), int(ay)), self.paddle_radius)
        
        # 畫玩家(藍色)
        px, py = self.agent_paddle.body.position
        pygame.draw.circle(self.screen, (0, 0, 255), (int(px), int(py)), self.paddle_radius)
        
        # 畫邊框
        pygame.draw.rect(self.screen, (0,0,0), (0,0,self.width, self.height), 5)
        
        # 畫球門線
        gw = self.goal_width
        pygame.draw.line(self.screen, (255,255,255), (self.width/2 - gw/2, 0), (self.width/2 + gw/2, 0), 5)
        pygame.draw.line(self.screen, (255,255,255), (self.width/2 - gw/2, self.height), (self.width/2 + gw/2, self.height), 5)
        
        pygame.display.flip()   # 使畫面更新
        self.clock.tick(self.metadata["render_fps"])    # 控制畫面最多60FPS

    def close(self):
        """關閉視窗與資源釋放"""
        # 若視窗已建立，關閉它
        if self.screen:
            pygame.quit()
            self.screen = None