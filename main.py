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
    satellite = RigidBody(mass=KG_TO_MU * 500, position=np.array([r, 0.0]), velocity=np.array([0.0, v]))

    engine.add_body(earth)
    engine.add_body(satellite)
    
    engine.initialize() # ベルレ法の初期化

    # ==========================================
    # View（カメラ）のセットアップ
    # ==========================================
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_DU)

    # ==========================================
    # ゲームループ（Controller & View）
    # ==========================================
    running = True
    while running:
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 物理状態の更新
        engine.step()

        # 画面のクリア
        screen.fill(COLOR_BG)

        # 描画
        # 地球の描画（地球半径は 1 DU なので，カメラのスケールをそのままピクセル半径として使う．）
        earth_screen_pos = camera.world_to_screen(earth.position)
        pygame.draw.circle(screen, COLOR_EARTH, earth_screen_pos, int(1.0 * camera.pixels_per_du))

        # 衛星の描画（画面上で見やすいように固定の5ピクセルで描画）
        sat_screen_pos = camera.world_to_screen(satellite.position)
        pygame.draw.circle(screen, COLOR_SAT, sat_screen_pos, 5)

        # 画面の更新
        pygame.display.flip()
        
        # フレームレートの制御
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
