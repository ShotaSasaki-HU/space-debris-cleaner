# main.py
import pygame
import sys
import numpy as np
import math

from physics.engine import GravityEngine
from physics.body import RigidBody
from physics.constants import KG_TO_MU, EARTH_MASS_KG, METER_TO_DU, EARTH_RADIUS_M, G_CANONICAL
from physics.control import PIDController

from view.camera import Camera, RelativeCamera
from view.renderer import GameRenderer

# アプリケーション全体の設定
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
PIXELS_PER_DU = 100.0
TIME_STEP_TU = 0.01

THRUST_POWER = 0.05 * (KG_TO_MU * 500)
TORQUE_POWER = 1.0 * 1.0

class SpaceDebrisApp:
    """
    ゲームアプリケーションのメインクラス．
    初期化，イベント処理，メインループの制御を行う．
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Space Debris Cleaner")
        self.clock = pygame.time.Clock()
        self.running = True
        self.view_mode = "MACRO" # "MACRO" または "MICRO"

        self._setup_physics()
        self._setup_view()
        self._setup_controls()

    def _setup_physics(self):
        """物理エンジンと天体の初期配置"""
        self.engine = GravityEngine(time_step=TIME_STEP_TU)
        M_earth = KG_TO_MU * EARTH_MASS_KG
        
        self.earth = RigidBody(mass=M_earth, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), is_fixed=True)
        
        r_player = METER_TO_DU * (EARTH_RADIUS_M + 10010e3)
        v_player = np.sqrt(G_CANONICAL * M_earth / r_player)
        self.player_sat = RigidBody(
            mass=KG_TO_MU * 500, position=np.array([r_player, 0.0]),
            velocity=np.array([0.0, v_player]), moment_of_inertia=1.0, angle=math.pi / 2.0
        )

        r_debris = METER_TO_DU * (EARTH_RADIUS_M + 10000e3)
        v_debris = np.sqrt(G_CANONICAL * M_earth / r_debris)
        self.target_debris = RigidBody(
            mass=KG_TO_MU * 500, position=np.array([r_debris, 0.0]),
            velocity=np.array([0.0, v_debris]), moment_of_inertia=1.0, angle=math.pi / 4.0
        )

        self.engine.add_body(self.earth)
        self.engine.add_body(self.player_sat)
        self.engine.add_body(self.target_debris)
        self.engine.initialize()

    def _setup_view(self):
        """描画関連の初期化"""
        self.macro_camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU) # ランデブー画面用
        self.micro_camera = RelativeCamera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU * 100) # 近傍運用画面用
        self.micro_camera.set_target(self.target_debris)
        self.renderer = GameRenderer(self.screen, self.macro_camera)

    def _setup_controls(self):
        """入力制御系の初期化"""
        self.sas_enabled = False
        self.sas_controller = PIDController(kp=1.0, ki=0.0, kd=1.0)

    def handle_events(self):
        """ユーザー入力の処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.sas_enabled = not self.sas_enabled
                # Enterキーでカメラを切り替える処理
                elif event.key == pygame.K_RETURN:
                    if self.view_mode == "MACRO":
                        self.view_mode = "MICRO"
                        self.renderer.camera = self.micro_camera
                    else:
                        self.view_mode = "MACRO"
                        self.renderer.camera = self.macro_camera
        
        keys = pygame.key.get_pressed()

        # 並進スラスター
        thrust_x, thrust_y = 0.0, 0.0
        if keys[pygame.K_w]: thrust_x += THRUST_POWER
        if keys[pygame.K_s]: thrust_x -= THRUST_POWER
        if keys[pygame.K_a]: thrust_y += THRUST_POWER
        if keys[pygame.K_d]: thrust_y -= THRUST_POWER

        if thrust_x != 0 or thrust_y != 0:
            self.player_sat.apply_local_force(thrust_x, thrust_y)
        
        # 回転制御
        if self.sas_enabled:
            target_angle = math.atan2(self.player_sat.velocity[1], self.player_sat.velocity[0])
            auto_torque = self.sas_controller.compute_torque(
                current_angle=self.player_sat.angle,
                target_angle=target_angle,
                dt_tu=TIME_STEP_TU
            )
            auto_torque = np.clip(auto_torque, -TORQUE_POWER, TORQUE_POWER)
            self.player_sat.apply_torque(auto_torque)
        else:
            if keys[pygame.K_q]: self.player_sat.apply_torque(TORQUE_POWER)
            if keys[pygame.K_e]: self.player_sat.apply_torque(-TORQUE_POWER)

    def update(self):
        """状態の更新"""
        self.engine.step()
        self.orbital_predictions = self.engine.predict_trajectories(
            future_duration=30.0, dt_prediction=0.05
        )

    def render(self):
        """画面の描画"""
        self.renderer.clear()
        self.renderer.draw_predictions(self.orbital_predictions, player=self.player_sat)
        self.renderer.draw_bodies(self.earth, self.player_sat, self.target_debris)
        self.renderer.draw_ui(self.player_sat, self.target_debris, self.sas_enabled)
        
        pygame.display.flip()

    def run(self):
        """メインループ"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = SpaceDebrisApp()
    app.run()
