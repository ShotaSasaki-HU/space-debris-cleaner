# main.py
import pygame
import sys
import numpy as np
import math

from physics.engine import GravityEngine
from physics.body import RigidBody
from physics.constants import KG_TO_MU, EARTH_MASS_KG, METER_TO_DU, EARTH_RADIUS_M, G_CANONICAL, SEC_TO_TU
from view.camera import Camera
from physics.control import PDController

# ==========================================
# 定数設定（描画用）
# ==========================================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
PIXELS_PER_DU = 100.0 # 1 DU（地球半径）を 100 ピクセルとして描画

COLOR_BG = (10, 10, 20)      # 宇宙の背景色（暗い紺色）
COLOR_EARTH = (50, 150, 255) # 地球の色
COLOR_SAT = (255, 200, 50)   # 衛星の色
COLOR_PREDICTION = (255, 255, 255, 150) # 予測線の色

THRUST_POWER = 0.05 * (KG_TO_MU * 500) # 並進推力
TORQUE_POWER = 1.0 * 1.0 # 回転トルク

def draw_satellite(screen: pygame.Surface, camera: Camera, body: RigidBody, color: tuple, size_du: float = 0.05):
    """
    衛星を現在の角度に基づき，進行方向が分かる三角形で描画する．
    """
    # ワールド座標系での三角形の3頂点のオフセットを計算
    # 機首（前）
    nose_offset = np.array([math.cos(body.angle), math.sin(body.angle)]) * size_du
    # 左後ろ (機首から140度ずらした位置)
    left_offset = np.array([math.cos(body.angle + 2.44), math.sin(body.angle + 2.44)]) * size_du
    # 右後ろ (機首から-140度ずらした位置)
    right_offset = np.array([math.cos(body.angle - 2.44), math.sin(body.angle - 2.44)]) * size_du

    # ワールド座標の絶対位置に変換
    nose_pos = body.position + nose_offset
    left_pos = body.position + left_offset
    right_pos = body.position + right_offset

    points = [
        camera.world_to_screen(nose_pos),
        camera.world_to_screen(left_pos),
        camera.world_to_screen(right_pos)
    ]

    pygame.draw.polygon(screen, color, points)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Space Debris Cleaner")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # ==========================================
    # Model（物理エンジン）のセットアップ
    # ==========================================
    # ゲームループの1フレームで進めるシミュレーション時間
    # 早送りをしたい場合はこの time_step を大きくする．
    engine = GravityEngine(time_step=0.01)

    # 地球の配置（DU空間）
    M_earth = KG_TO_MU * EARTH_MASS_KG
    earth = RigidBody(mass=M_earth, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), is_fixed=True)
    
    # クリーナー衛星
    r = METER_TO_DU * (EARTH_RADIUS_M + 400e3)
    v = np.sqrt(G_CANONICAL * M_earth / r)
    player_sat = RigidBody(
        mass=KG_TO_MU * 500, 
        position=np.array([r, 0.0]), 
        velocity=np.array([0.0, v]),
        moment_of_inertia=1.0,
        angle=math.pi / 2.0 
    )

    r = METER_TO_DU * (EARTH_RADIUS_M + 10000e3)
    # v = 10e3 * METER_TO_DU / SEC_TO_TU
    v = np.sqrt(G_CANONICAL * M_earth / r)
    sat2 = RigidBody(
        mass=KG_TO_MU * 500, 
        position=np.array([r, 0.0]), 
        velocity=np.array([0.0, v]),
        moment_of_inertia=1.0,
        angle=math.pi / 2.0 
    )

    engine.add_body(earth)
    engine.add_body(player_sat)
    engine.add_body(sat2)
    
    engine.initialize() # ベルレ法の初期化

    # ==========================================
    # View（カメラ）のセットアップ
    # ==========================================
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU)

    # ==========================================
    # ゲームループ（Controller & View）
    # ==========================================

    # フライトコンピュータSASの初期設定
    sas_enabled = False
    sas_controller = PDController(kp=4.0, kd=1.0)

    # 予測線描画用サーフェス（透明度を使用するため，SRCALPHAフラグが必要．）
    prediction_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    running = True
    while running:
        # ==========================================
        # Controller層（ユーザー入力処理）
        # ==========================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    sas_enabled = not sas_enabled # SASのトグル
        
        keys = pygame.key.get_pressed()

        # 並進スラスター
        thrust_x = 0.0
        thrust_y = 0.0
        if keys[pygame.K_w]: thrust_x += THRUST_POWER
        if keys[pygame.K_s]: thrust_x -= THRUST_POWER
        if keys[pygame.K_a]: thrust_y += THRUST_POWER
        if keys[pygame.K_d]: thrust_y -= THRUST_POWER

        if thrust_x != 0 or thrust_y != 0:
            player_sat.apply_local_force(thrust_x, thrust_y)
        
        if sas_enabled:
            # SAS ON: 進行方向に自動で指向
            target_angle = math.atan2(player_sat.velocity[1], player_sat.velocity[0])
            
            auto_torque = sas_controller.compute_torque(
                current_angle=player_sat.angle,
                target_angle=target_angle,
                current_angular_velocity=player_sat.angular_velocity
            )
            auto_torque = np.clip(auto_torque, -TORQUE_POWER, TORQUE_POWER)

            player_sat.apply_torque(auto_torque)
        else:
            # SAS OFF: 手動操作（QE）
            if keys[pygame.K_q]: player_sat.apply_torque(TORQUE_POWER) # CCW
            if keys[pygame.K_e]: player_sat.apply_torque(-TORQUE_POWER) # CW

        # ==========================================
        # Model層（状態の更新と予測）
        # ==========================================
        engine.step()

        # 軌道予測の計算
        # パフォーマンスが悪い場合は，呼び出し頻度やデルタタイムを下げる．
        orbital_predictions = engine.predict_trajectories(future_duration=2.0 * np.pi * 4.0, dt_prediction=0.05)

        # ==========================================
        # View層（描画）
        # ==========================================
        screen.fill(COLOR_BG) # 画面のクリア
        prediction_surface.fill((0,0,0,0)) # 透明

        # 予測線の描画
        for body_id, path in orbital_predictions.items():
            # 対応するRigidBodyを探す．
            body = next((b for b in engine.bodies if id(b) == body_id), None)
            if body is None or body.is_fixed:
                continue
            
            # DU座標リストをピクセル座標リストに変換
            if len(path) < 2:
                continue
            screen_points = [camera.world_to_screen(p) for p in path]
            
            # アンチエイリアス付きの連続線を描画
            # AALinesは透明度に対応していないため，一旦専用サーフェスに描画する．
            pygame.draw.aalines(prediction_surface, COLOR_PREDICTION, False, screen_points)

        # 予測線サーフェスをメイン画面に合成
        screen.blit(prediction_surface, (0, 0))

        # オブジェクトの描画
        # 地球の描画（地球半径は 1 DU なので，カメラのスケールをそのままピクセル半径として使う．）
        earth_screen_pos = camera.world_to_screen(earth.position)
        pygame.draw.circle(screen, COLOR_EARTH, earth_screen_pos, int(1.0 * camera.pixels_per_du))

        # 衛星の描画
        draw_satellite(screen, camera, player_sat, (255, 0, 0), size_du=0.08)
        draw_satellite(screen, camera, sat2, COLOR_SAT, size_du=0.08)
        
        # SASステータス
        sas_text = "SAS: ON" if sas_enabled else "SAS: OFF"
        sas_color = (100, 255, 100) if sas_enabled else (200, 200, 200)
        text_surface = font.render(sas_text, True, sas_color)
        screen.blit(text_surface, (20, 20))
        
        # 操作説明
        help_text1 = font.render("T: Toggle SAS | W/S: Forward/Backward | A/D: Left/Right", True, (150, 150, 150))
        help_text2 = font.render("Q/E: Manual Rotation (when SAS is OFF)", True, (150, 150, 150))
        screen.blit(help_text1, (20, 50))
        screen.blit(help_text2, (20, 70))

        pygame.display.flip() # 画面の更新
        clock.tick(FPS) # フレームレートの制御

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
