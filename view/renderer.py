# view/renderer.py
import pygame
import numpy as np
import math

from physics.body import RigidBody
from view.camera import Camera
from physics.constants import METER_TO_DU, SEC_TO_TU

COLOR_EARTH = (50, 150, 255)
COLOR_PLAYER = (0, 255, 0)
COLOR_DEBRIS = (200, 200, 200)
COLOR_PREDICTION = (255, 255, 255, 150)
COLOR_UI_TEXT = (220, 220, 220)

class GameRenderer:
    """
    ゲーム画面の描画を統括するクラス．UIやオブジェクトの描画ロジックをカプセル化する．
    """
    def __init__(self, screen: pygame.Surface, camera: Camera):
        self.screen = screen
        self.camera = camera
        self.font = pygame.font.SysFont("couriernew", 20)
        
        # 予測線描画用の透明サーフェス
        self.prediction_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    def clear(self, bg_color: tuple = (10, 10, 20)):
        """画面をクリアする"""
        self.screen.fill(bg_color)
        self.prediction_surface.fill((0, 0, 0, 0))

    def draw_predictions(self, predictions_dict: dict, player: RigidBody):
        """全オブジェクトの軌道予測線を描画する"""
        for body_id, path in predictions_dict.items():
            if len(path) < 2: continue
            
            screen_points = [self.camera.world_to_screen(p) for p in path]
            if body_id == id(player):
                pygame.draw.aalines(self.prediction_surface, COLOR_PLAYER + (150, ), False, screen_points)
            else:
                pygame.draw.aalines(self.prediction_surface, COLOR_PREDICTION, False, screen_points)
            
        self.screen.blit(self.prediction_surface, (0, 0))

    def draw_bodies(self, earth: RigidBody, player: RigidBody, debris: RigidBody):
        """宇宙の天体・オブジェクトを描画する"""
        # 地球
        earth_pos = self.camera.world_to_screen(earth.position)
        pygame.draw.circle(self.screen, COLOR_EARTH, earth_pos, int(1.0 * self.camera.pixels_per_du))

        # デブリ
        debris_size_du = 12.0 / self.camera.pixels_per_du
        self._draw_debris(debris, COLOR_DEBRIS, size_du=debris_size_du)

        # プレイヤー
        # 画面上でNピクセルの大きさになるように，現在のカメラのスケールから DU を逆算する．
        player_size_du = 8.0 / self.camera.pixels_per_du
        self._draw_satellite(player, COLOR_PLAYER, size_du=player_size_du)

    def _draw_debris(self, body: RigidBody, color: tuple, size_du: float):
        """デブリを回転する四角形で描画する内部メソッド"""
        half_s = size_du / 2.0
        # ローカル座標系での4つの頂点
        corners = [
            np.array([ half_s,  half_s]),
            np.array([-half_s,  half_s]),
            np.array([-half_s, -half_s]),
            np.array([ half_s, -half_s])
        ]
        
        cos_t = math.cos(body.angle)
        sin_t = math.sin(body.angle)
        
        points = []
        for c in corners:
            # 回転行列でローカル座標を回転
            rx = c[0] * cos_t - c[1] * sin_t
            ry = c[0] * sin_t + c[1] * cos_t
            # ワールド座標に変換し，カメラに通す．
            world_pos = body.position + np.array([rx, ry])
            points.append(self.camera.world_to_screen(world_pos))
            
        pygame.draw.polygon(self.screen, color, points)

    def _draw_satellite(self, body: RigidBody, color: tuple, size_du: float):
        """衛星を三角形で描画する内部メソッド"""
        nose_offset = np.array([math.cos(body.angle), math.sin(body.angle)]) * size_du
        left_offset = np.array([math.cos(body.angle + 2.44), math.sin(body.angle + 2.44)]) * size_du
        right_offset = np.array([math.cos(body.angle - 2.44), math.sin(body.angle - 2.44)]) * size_du

        points = [
            self.camera.world_to_screen(body.position + nose_offset),
            self.camera.world_to_screen(body.position + left_offset),
            self.camera.world_to_screen(body.position + right_offset)
        ]
        pygame.draw.polygon(self.screen, color, points)

    def draw_ui(self, player: RigidBody, target: RigidBody, sas_enabled: bool):
        """各種UIを描画する"""
        self._draw_rel_nav_ui(player, target)
        self._draw_control_console(sas_enabled)

    def _draw_rel_nav_ui(self, player: RigidBody, target: RigidBody):
        """相対ナビゲーションUI"""
        rel_pos_du = target.position - player.position
        rel_vel_du_tu = target.velocity - player.velocity
        
        dist_m = np.linalg.norm(rel_pos_du) / METER_TO_DU
        rel_speed_m_s = (np.linalg.norm(rel_vel_du_tu) / METER_TO_DU) * SEC_TO_TU
        
        is_approaching = np.dot(rel_pos_du, rel_vel_du_tu) < 0
        approach_sign = "-" if is_approaching else "+"

        ui_lines = [
            f"--- TARGET NAV DATA ---",
            f"Relative Dist : {dist_m:,.1f} m",
            f"Relative Vel  : {approach_sign}{rel_speed_m_s:.2f} m/s",
        ]
        
        y_offset = self.screen.get_height() - 90
        for i, line in enumerate(ui_lines):
            color = (255, 150, 150) if "Vel" in line and is_approaching else COLOR_UI_TEXT
            text_surf = self.font.render(line, True, color)
            self.screen.blit(text_surf, (20, y_offset + i * 22))

    def _draw_control_console(self, sas_enabled: bool):
        """操作に関するテキスト表示"""
        # UIの一番上に現在のカメラモードを描画
        mode_text = "VIEW: " + ("MACRO (Absolute)" if isinstance(self.camera, Camera) else "MICRO/NANO (Relative)")
        self.screen.blit(self.font.render(mode_text, True, COLOR_UI_TEXT), (20, 20))
        
        sas_text = "SAS: ON" if sas_enabled else "SAS: OFF"
        sas_color = (100, 255, 100) if sas_enabled else (200, 200, 200)
        self.screen.blit(self.font.render(sas_text, True, sas_color), (20, 40))
        
        help_color = (150, 150, 150)
        self.screen.blit(self.font.render("W/S: Forward/Backward | A/D: Left/Right", True, help_color), (20, 60))
        self.screen.blit(self.font.render("Q/E: Manual Rotation (SAS OFF) | T: Toggle SAS", True, help_color), (20, 80))
