# main.py
import pygame
import sys
import numpy as np

from physics.engine import GravityEngine
from physics.body import RigidBody
from physics.constants import KG_TO_MU, EARTH_MASS_KG, METER_TO_DU, EARTH_RADIUS_M, G_CANONICAL, SEC_TO_TU
from view.camera import Camera

# ==========================================
# 定数設定（描画用）
# ==========================================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# 1 DU（地球半径）を 100 ピクセルとして描画
PIXELS_PER_DU = 100.0

# 色の定義 (R, G, B)
COLOR_BG = (10, 10, 20)      # 宇宙の背景色（暗い紺色）
COLOR_EARTH = (50, 150, 255) # 地球の色
COLOR_SAT = (255, 200, 50)   # 衛星の色
COLOR_PREDICTION = (255, 255, 255, 150) # 予測線の色

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Space Debris Cleaner")
    clock = pygame.time.Clock()

    # ==========================================
    # Model（物理エンジン）のセットアップ
    # ==========================================
    # ゲームループの1フレームで進めるシミュレーション時間
    # 早送りをしたい場合はこの time_step を大きくする．
    engine = GravityEngine(time_step=0.05)

    # 地球の配置（DU空間）
    M_earth = KG_TO_MU * EARTH_MASS_KG
    earth = RigidBody(mass=M_earth, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), is_fixed=True)
    
    # 衛星の配置
    r = METER_TO_DU * (EARTH_RADIUS_M + 400e3)
    # v = np.sqrt(G_CANONICAL * M_earth / r)
    v = 10e3 * METER_TO_DU / SEC_TO_TU
    satellite1 = RigidBody(mass=KG_TO_MU * 500, position=np.array([r, 0.0]), velocity=np.array([0.0, v]))

    r = METER_TO_DU * (EARTH_RADIUS_M + 400e3)
    v = np.sqrt(G_CANONICAL * M_earth / r)
    satellite2 = RigidBody(mass=KG_TO_MU * 500, position=np.array([r, 0.0]), velocity=np.array([0.0, v]))

    engine.add_body(earth)
    engine.add_body(satellite1)
    engine.add_body(satellite2)
    
    engine.initialize() # ベルレ法の初期化

    # ==========================================
    # View（カメラ）のセットアップ
    # ==========================================
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU)

    # ==========================================
    # ゲームループ（Controller & View）
    # ==========================================

    # 予測線描画用サーフェス（透明度を使用するため，SRCALPHAフラグが必要．）
    prediction_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    running = True
    while running:
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 物理状態の更新
        engine.step()

        # 軌道予測の計算
        # パフォーマンスが悪い場合は，呼び出し頻度やデルタタイムを下げる．
        orbital_predictions = engine.predict_trajectories(future_duration=2.0 * np.pi * 4.0, dt_prediction=0.05)

        # 画面のクリア
        screen.fill(COLOR_BG)
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

        # 衛星の描画（画面上で見やすいように固定の5ピクセルで描画）
        for b in engine.bodies:
            if b.is_fixed: continue
            sat_screen_pos = camera.world_to_screen(b.position)
            pygame.draw.circle(screen, COLOR_SAT, sat_screen_pos, 5)

        pygame.display.flip() # 画面の更新
        clock.tick(FPS) # フレームレートの制御

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
