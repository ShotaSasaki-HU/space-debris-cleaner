# main.py
import pygame
import sys
import numpy as np
from datetime import datetime, timedelta, timezone

from physics.engine import GravityEngine
from physics.body import RigidBody
from physics.constants import (
    KG_TO_MU, EARTH_MASS_KG, METER_TO_DU, EARTH_RADIUS_M, G_CANONICAL, TU_TO_SEC, SEC_TO_TU,
    CLEANER_SAT_MASS_KG, CLEANER_SAT_MOMENT_OF_INERTIA_KG_M2, CLEANER_SAT_SIZE_METER,
    MAX_THRUST_NEWTON, MAX_TORQUE_NM, NEWTON_TO_CANONICAL, NM_TO_CANONICAL
)
from physics.control import PIDController
from view.camera import Camera, RelativeCamera
from view.renderer import GameRenderer

# アプリケーション全体の設定
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
PIXELS_PER_DU = 100.0
TIME_STEP_TU_PHYSICS = (1 / FPS) * SEC_TO_TU # 物理エンジンの微小ステップ幅

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

        self.view_mode = "MACRO"
        self.throttle = 1.0 # スロットル100%

        self.fast_forward_rate = 1.0 # 早送り倍率
        self.time_accumulator = 0.0 # 未処理のシミュレーション時間を貯めるバケツ

        self.simulation_time = datetime.now(timezone.utc) # ゲーム内の時刻（物理演算には関係しない．）
        self.mission_start_time = self.simulation_time

        self._setup_physics()
        self._setup_view()
        self._setup_controls()

    def _setup_physics(self):
        """物理エンジンと天体の初期配置"""
        self.engine = GravityEngine(time_step=TIME_STEP_TU_PHYSICS)
        M_earth = KG_TO_MU * EARTH_MASS_KG
        self.earth = RigidBody(mass=M_earth, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), is_fixed=True)
        
        r_player = METER_TO_DU * (EARTH_RADIUS_M + 10010e3)
        v_player = np.sqrt(G_CANONICAL * M_earth / r_player)
        m_sat_cano = CLEANER_SAT_MASS_KG * KG_TO_MU
        i_sat_cano = CLEANER_SAT_MOMENT_OF_INERTIA_KG_M2 * KG_TO_MU * (METER_TO_DU ** 2)
        self.player_sat = RigidBody(
            mass=m_sat_cano,
            position=np.array([r_player, 0.0]),
            velocity=np.array([0.0, v_player]),
            moment_of_inertia=i_sat_cano,
            angle=np.pi / 2.0,
            image_path="assets/images/player_sat.png",
            real_width_du=CLEANER_SAT_SIZE_METER[0] * METER_TO_DU,
            real_height_du=CLEANER_SAT_SIZE_METER[1] * METER_TO_DU,
            draw_fixed_size_px=30
        )

        r_2nd_stage = METER_TO_DU * (EARTH_RADIUS_M + 10000e3)
        v_2nd_stage = np.sqrt(G_CANONICAL * M_earth / r_2nd_stage)
        m_2nd_stage_cano = 3000 * KG_TO_MU # H-IIAロケット15号機の上段
        i_2nd_stage_cano = 33250 * KG_TO_MU * (METER_TO_DU ** 2) # 円柱の中心軸に垂直な軸まわりの慣性モーメント
        self.target_debri = RigidBody(
            mass=m_2nd_stage_cano,
            position=np.array([r_2nd_stage, 0.0]),
            velocity=np.array([0.0, v_2nd_stage]),
            moment_of_inertia=i_2nd_stage_cano,
            angle=0.0,
            image_path="assets/images/rocket_2nd_stage.png",
            real_width_du=11.0 * METER_TO_DU,
            real_height_du=4.0 * METER_TO_DU,
            draw_fixed_size_px=30
        )

        self.engine.add_body(self.earth)
        self.engine.add_body(self.player_sat)
        self.engine.add_body(self.target_debri)
        self.engine.initialize()

        self.max_thrust_cano = MAX_THRUST_NEWTON * NEWTON_TO_CANONICAL
        self.max_torque_cano = MAX_TORQUE_NM * NM_TO_CANONICAL

    def _setup_view(self):
        """描画関連の初期化"""
        self.macro_camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU)

        self.micro_camera = RelativeCamera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU * 100)
        self.micro_camera.set_target(self.target_debri)

        self.nano_camera = RelativeCamera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU * 10000)
        self.nano_camera.set_target(self.target_debri)

        self.renderer = GameRenderer(self.screen, self.macro_camera)

    def _setup_controls(self):
        """入力制御系の初期化"""
        self.sas_enabled = False
        self.sas_controller = PIDController(kp=0.5, ki=0.0, kd=5.0)

    def handle_events(self):
        """ユーザー入力の処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.sas_enabled = not self.sas_enabled
                # カメラの3段階切り替えロジック（エンターキー）
                elif event.key == pygame.K_RETURN:
                    if self.view_mode == "MACRO":
                        self.view_mode = "MICRO"
                        self.renderer.camera = self.micro_camera
                    elif self.view_mode == "MICRO":
                        self.view_mode = "NANO"
                        self.renderer.camera = self.nano_camera
                    else:
                        self.view_mode = "MACRO"
                        self.renderer.camera = self.macro_camera
                # 早送り係数の操作
                elif event.key == pygame.K_PERIOD:
                    self.fast_forward_rate = min(1000.0, self.fast_forward_rate * 10.0)
                elif event.key == pygame.K_COMMA:
                    self.fast_forward_rate = max(1.0, self.fast_forward_rate / 10.0)
        
        keys = pygame.key.get_pressed()

        # スロットル（_apply_control_forcesの中に書くと爆速で上下してしまうためココに書く．）
        if keys[pygame.K_UP]:
            self.throttle = min(1.0, self.throttle + 0.01)
        if keys[pygame.K_DOWN]:
            self.throttle = max(0.01, self.throttle - 0.01)
    
    def _apply_control_forces(self, dt_tu: float):
        """ユーザによる並進・回転およびSAS（安定増大装置）の計算"""
        keys = pygame.key.get_pressed()

        # 並進スラスター
        thrust_mag = self.max_thrust_cano * self.throttle
        thrust_x, thrust_y = 0.0, 0.0
        if keys[pygame.K_w]: thrust_x += thrust_mag
        if keys[pygame.K_s]: thrust_x -= thrust_mag
        if keys[pygame.K_a]: thrust_y += thrust_mag
        if keys[pygame.K_d]: thrust_y -= thrust_mag

        if thrust_x != 0 or thrust_y != 0:
            self.player_sat.apply_local_force(thrust_x, thrust_y)
        
        # 回転制御
        self.player_torque = 0.0
        if self.sas_enabled:
            target_angle = np.atan2(self.player_sat.velocity[1], self.player_sat.velocity[0])
            
            omega_si = self.player_sat.angular_velocity / TU_TO_SEC
            dt_si = dt_tu * TU_TO_SEC

            # フライトコンピュータは馴染みのあるSI単位系で計算
            auto_torque_nm = self.sas_controller.compute_torque(current_angle=self.player_sat.angle, target_angle=target_angle,
                                                                current_angular_velocity=omega_si, dt_sec=dt_si)
            
            # 得られたトルク（N·m）をカノニカル単位系に変換して物理エンジンに渡す．
            auto_torque_cano = auto_torque_nm * NM_TO_CANONICAL
            self.player_torque = np.clip(auto_torque_cano, -self.max_torque_cano, self.max_torque_cano)
        else:
            if keys[pygame.K_q]: self.player_torque = self.max_torque_cano * self.throttle
            if keys[pygame.K_e]: self.player_torque = -self.max_torque_cano * self.throttle
        self.player_sat.apply_torque(self.player_torque)

    def update(self, dt_real_sec: float):
        """
        Args:
            dt_real_sec (float): 前フレームから経過した現実の時間（秒）
        """
        self.time_accumulator += dt_real_sec * SEC_TO_TU * self.fast_forward_rate

        if self.fast_forward_rate >= 1000:
            current_physics_dt_tu = TIME_STEP_TU_PHYSICS * 10**(int(np.log10(self.fast_forward_rate)) - 2)
        else:
            current_physics_dt_tu = TIME_STEP_TU_PHYSICS

        while self.time_accumulator >= current_physics_dt_tu:
            self.engine.time_step = current_physics_dt_tu # エンジン内部の時間幅を動的に書き換える．

            self._apply_control_forces(dt_tu=current_physics_dt_tu) # ユーザによる並進・回転入力の評価
            self.engine.step()
            self.time_accumulator -= current_physics_dt_tu

            self.simulation_time += timedelta(seconds=current_physics_dt_tu * TU_TO_SEC) # ループの外でもほぼ問題ないが，厳密を期すならココ．

        # 軌道予測は重いため，1フレームに1回だけ実行する．
        self.orbital_predictions = self.engine.predict_trajectories(future_duration=30.0, dt_prediction=0.05)

    def render(self):
        """画面の描画"""
        self.renderer.clear()
        self.renderer.draw_predictions(self.orbital_predictions, player=self.player_sat)
        self.renderer.draw_bodies(self.earth, self.player_sat, self.target_debri)
        self.renderer.draw_ui(self.player_sat, self.target_debri, self.sas_enabled, self.throttle,
                              self.player_torque, self.mission_start_time, self.simulation_time, self.fast_forward_rate)
        pygame.display.flip()

    def run(self):
        """メインループ"""
        while self.running:
            dt_real_sec = self.clock.tick(FPS) / 1000.0 # FPSの上限を設定し，前回からの経過時間を返す．
            self.handle_events()
            self.update(dt_real_sec)
            self.render()
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = SpaceDebrisApp()
    app.run()
