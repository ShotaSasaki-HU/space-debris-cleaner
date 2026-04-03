# view/renderer.py
import pygame
import numpy as np
from typing import Dict
from datetime import datetime
from skyfield.api import load, Star
from skyfield.data import hipparcos

from physics.body import RigidBody
from view.camera import Camera, RelativeCamera
from physics.constants import (
    METER_TO_DU, SEC_TO_TU, TU_TO_SEC, MAX_THRUST_NEWTON, MAX_TORQUE_NM, NM_TO_CANONICAL, KG_TO_MU
)

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
        self.image_cache: Dict[str, pygame.Surface] = {}
        self.font = pygame.font.SysFont("couriernew", 20)
        self.prediction_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        self.scaled_cache: Dict[str, tuple[int, int, pygame.Surface]] = {}

        self._setup_starry_sky()

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

    def draw_bodies(self, bodies: list[RigidBody], selected_body: RigidBody):
        """宇宙の天体・オブジェクトを描画する"""
        for body in bodies:
            is_selected = (body is selected_body)
            self._draw_realistic_body(body=body, is_selected=is_selected)

    def _draw_realistic_body(self, body: RigidBody, is_selected: bool = False):
        """画像をロードし，視点に合わせてスケーリング・回転して描画する．"""
        if body.image_path:
            if body.image_path not in self.image_cache:
                try:
                    image = pygame.image.load(body.image_path).convert_alpha() # convert_alpha()で透過を有効にする．
                    self.image_cache[body.image_path] = image
                except pygame.error as e:
                    print(f"Error loading image: {body.image_path}, {e}")
                    body.image_path = None # ロードに失敗した場合は，画像パスをNoneにしてフォールバックする．

            image: pygame.Surface = self.image_cache.get(body.image_path)
            if image:
                # スケーリングロジック
                # まずは，原寸大でスケーリングしてみる．
                target_w = int(body.real_width_du * self.camera.pixels_per_du)
                target_h = int(body.real_height_du * self.camera.pixels_per_du)

                # 原寸大が小さすぎたら，固定サイズに変更する．
                if min(target_w, target_h) < body.draw_fixed_size_px:
                    orig_w, orig_h = image.get_size()
                    target_size_px = body.draw_fixed_size_px
                    if orig_w > orig_h:
                        target_w = target_size_px
                        target_h = int(orig_h * target_size_px / orig_w)
                    else:
                        target_h = target_size_px
                        target_w = int(orig_w * target_size_px / orig_h)
                
                # 厳密性に欠けるが，大きすぎる画像はクラッシュを起こすため放棄する．
                if target_w > 5000 or target_h > 5000: return

                # --- オフセット適用ココカラ ---

                visual_offset_world = np.array([0.0, 0.0])
                if hasattr(body, 'visual_offset_local') and np.any(body.visual_offset_local):
                    cos_b = np.cos(body.angle)
                    sin_b = np.sin(body.angle)
                    lx, ly = body.visual_offset_local
                    visual_offset_world = np.array([lx * cos_b - ly * sin_b, lx * sin_b + ly * cos_b])

                # 重心位置にオフセットを足した場所をスクリーンの中心とする
                body_pos_px = self.camera.world_to_screen(body.position + visual_offset_world) # 物理エンジンの座標を画面座標へ変換

                # --- オフセット適用ココマデ ---
                
                # --- カリングココカラ ---

                # 画像のだいたいの半径
                approx_radius = max(target_w, target_h) * 0.75
                screen_w, screen_h = self.screen.get_size()
                
                # 画面内に入っているか判定
                is_on_screen = not (
                    body_pos_px[0] + approx_radius < 0 or 
                    body_pos_px[0] - approx_radius > screen_w or
                    body_pos_px[1] + approx_radius < 0 or 
                    body_pos_px[1] - approx_radius > screen_h
                )

                if not is_on_screen: return

                # --- カリングココマデ ---

                # --- スケーリングのキャッシュココカラ ---

                cache_key = body.image_path
                needs_scaling = True

                # 前回と同じサイズならキャッシュを利用する．pygameのscale演算が重いため．
                if cache_key in self.scaled_cache:
                    cached_w, cached_h, cached_surf = self.scaled_cache[cache_key]
                    if cached_w == target_w and cached_h == target_h:
                        scaled_image = cached_surf
                        needs_scaling = False

                if needs_scaling:
                    # 大きな画像では，重いsmoothscaleの使用を避ける．
                    if target_w > 2000 or target_h > 2000:
                        scaled_image = pygame.transform.scale(image, (target_w, target_h))
                    else:
                        scaled_image = pygame.transform.smoothscale(image, (target_w, target_h))

                    self.scaled_cache[cache_key] = (target_w, target_h, scaled_image) # キャッシュを保存

                # --- スケーリングのキャッシュココマデ ---

                # Cameraの場合
                if isinstance(self.camera, Camera):
                    rotated_image = pygame.transform.rotate(scaled_image, np.rad2deg(body.angle))
                # RelativeCameraの場合
                else:
                    # ターゲットの「地球に対する角度（位相）」を計算
                    theta = np.atan2(self.camera.target_body.position[1], self.camera.target_body.position[0])
                    rotated_image = pygame.transform.rotate(scaled_image, np.rad2deg((np.pi / 2) - theta + body.angle))
                
                rotated_rect = rotated_image.get_rect() # 回転後の画像サイズを取得
                draw_pos = (body_pos_px[0] - rotated_rect.width // 2,
                            body_pos_px[1] - rotated_rect.height // 2) # 画像の中心座標を取得

                # 結合されているターゲットの再帰描画
                if getattr(body, 'docked_body', None):
                    docked = body.docked_body
                        
                    dx, dy = body.docked_offset_local
                    docked_offset_world = np.array([dx * cos_b - dy * sin_b, dx * sin_b + dy * cos_b])
                        
                    # 元のパラメータを退避
                    orig_pos = docked.position
                    orig_angle = docked.angle
                        
                    # 親の座標と角度に追従させて一時的に上書き
                    docked.position = body.position + docked_offset_world
                    docked.angle = body.angle + body.docked_rel_angle
                        
                    # 再帰描画（縁取りは消す）
                    self._draw_realistic_body(docked, is_selected=False)
                        
                    # 復元
                    docked.position = orig_pos
                    docked.angle = orig_angle
                
                # 選択されているbodyの縁取り
                if is_selected:
                    opaque_mask = pygame.mask.from_surface(rotated_image) # 画像の非透過部分からマスクを生成
                    outline = opaque_mask.outline() # マスクの輪郭座標リスト
                    if len(outline) >= 2:
                        # 輪郭座標は画像の左上が原点なので，画面描画座標 (draw_pos) にオフセットを加算する．
                        outline_points = [(p[0] + draw_pos[0], p[1] + draw_pos[1]) for p in outline]

                        # 縁取りの透過
                        temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                        pygame.draw.lines(temp_surf, (255, 255, 0, 255), True, outline_points, 2)
                        self.screen.blit(temp_surf, (0, 0))
                
                self.screen.blit(rotated_image, draw_pos) # ターゲットより後で前面に描画

                return # リアル画像での描画に成功したら終了

        # フォールバック（画像がない場合の描画）
        body_screen_pos = self.camera.world_to_screen(body.position)
        pygame.draw.rect(self.screen, (255, 255, 255), (body_screen_pos[0]-6, body_screen_pos[1]-6, 12, 12))

    def draw_ui(self, player: RigidBody, target: RigidBody, sas_enabled: bool, throttle: float, player_torque: float,
                mission_start_time: datetime, simulation_time: datetime, fast_forward_rate: float, capture_state: str, progress: float):
        """各種UIを描画する"""
        self._draw_control_console(sas_enabled, throttle, player, player_torque)
        self._draw_time(mission_start_time, simulation_time, fast_forward_rate)
        self._draw_capture_ui(capture_state, progress)
        self._draw_nav_data(player, target)
        self._draw_fuel_gage(player)

    def _draw_control_console(self, sas_enabled: bool, throttle: float, player: RigidBody, player_torque: float):
        """操作に関するテキスト表示"""

        """
        # UIの一番上に現在のカメラモードを描画
        mode_text = "VIEW: " + ("EARTH" if isinstance(self.camera, Camera) else "TRACKING")
        self.screen.blit(self.font.render(mode_text, True, COLOR_UI_TEXT), (20, 20))
        
        sas_text = "SAS: ON" if sas_enabled else "SAS: OFF"
        sas_color = (100, 255, 100) if sas_enabled else (200, 200, 200)
        self.screen.blit(self.font.render(sas_text, True, sas_color), (20, 40))
        """

        # スロットル
        self._draw_bar_gauge(screen=self.screen, cx=self.screen.get_width() - 85, cy=self.screen.get_height() - 105, w=50, h=150,
                             angle=0.0, min_val=0.0, max_val=1.0, input_val=throttle, full_color=(255, 0, 0),
                             stack_labels=['Throttle', f'{throttle * 100:.0f}%'], is_gradation=True)

        cx = 180
        cy = self.screen.get_height() - 160

        # --- スラスター動作状況ココカラ ---
        
        # プレイヤー画像の描画（常に上向き）
        if player.image_path and player.image_path in self.image_cache:
            image = self.image_cache[player.image_path]
            # UI用に固定サイズにスケーリング
            orig_w, orig_h = image.get_size()
            scale_factor = 60.0 / max(orig_w, orig_h)
            hud_img = pygame.transform.smoothscale(image, (int(orig_w * scale_factor), int(orig_h * scale_factor)))
            
            # 画像のデフォルト（右向き）を上向きに回転させて表示
            hud_img = pygame.transform.rotate(hud_img, 90)
            img_rect = hud_img.get_rect(center=(cx, cy))
            self.screen.blit(hud_img, img_rect.topleft)
        else:
            # フォールバックの三角形
            pygame.draw.polygon(self.screen, COLOR_PLAYER, [(cx, cy - 30), (cx - 20, cy + 20), (cx + 20, cy + 20)])

        keys = pygame.key.get_pressed()
        
        # 4方向のバーゲージの描画
        bar_w = 20
        bar_h = 50
        offset = 60 # 画像中心からの距離

        # 噴射時はスロットル値に比例，非噴射時は0．
        val_w = throttle if (keys[pygame.K_w] and player.propellant_mass > 0) else 0.0
        val_s = throttle if (keys[pygame.K_s] and player.propellant_mass > 0) else 0.0
        val_a = throttle if (keys[pygame.K_a] and player.propellant_mass > 0) else 0.0
        val_d = throttle if (keys[pygame.K_d] and player.propellant_mass > 0) else 0.0

        for theta, val_i in zip(np.arange(0, -2 * np.pi, -np.pi / 2), [val_a, val_s, val_d, val_w]):
            self._draw_bar_gauge(
                screen=self.screen,
                cx=cx + (offset * np.cos(theta)), # Y軸の正方向が下向きである事に注意
                cy=cy + (offset * np.sin(theta)),
                w=bar_w,
                h=bar_h,
                angle=-theta - (np.pi / 2),
                min_val=0.0,
                max_val=1.0,
                input_val=val_i,
                full_color=(255, 0, 0),
                stack_labels=['THR', f'{MAX_THRUST_NEWTON * val_i:.1f} N'],
                is_gradation=True
            )
        
        ## --- トルク表示ココカラ ---

        bar_w = 10
        bar_h = 2 * offset

        torque_nm = player_torque / NM_TO_CANONICAL
        val_ccw = max(0.0, torque_nm)
        val_cw = max(0.0, -torque_nm)

        # 正のトルク（CCW）
        self._draw_circular_gauge(
            screen=self.screen,
            cx=cx,
            cy=cy,
            radius=2.8 * offset,
            thickness=10,
            min_val=0.0,
            max_val=MAX_TORQUE_NM,
            input_val=val_ccw,
            full_color=(0, 255, 0),
            center_labels=[],
            start_angle_rad=np.deg2rad(90),
            end_angle_rad=np.deg2rad(150),
            is_flipped_horizontally=False,
            is_gradation=True
        )
        # 負のトルク（CW）
        self._draw_circular_gauge(
            screen=self.screen,
            cx=cx,
            cy=cy,
            radius=2.8 * offset,
            thickness=10,
            min_val=0.0,
            max_val=MAX_TORQUE_NM,
            input_val=val_cw,
            full_color=(0, 255, 0),
            center_labels=[],
            start_angle_rad=np.deg2rad(90),
            end_angle_rad=np.deg2rad(150),
            is_flipped_horizontally=True,
            is_gradation=True
        )

        torque_text = f'TRQ'
        torque_surf = self.font.render(torque_text, True, COLOR_UI_TEXT)
        torque_rect = torque_surf.get_rect(center=(cx, cy - offset - 142))
        self.screen.blit(torque_surf, torque_rect.topleft)

        torque_text = f'{torque_nm:.3f} N·m'
        torque_surf = self.font.render(torque_text, True, COLOR_UI_TEXT)
        torque_rect = torque_surf.get_rect(center=(cx, cy - offset - 120))
        self.screen.blit(torque_surf, torque_rect.topleft)

        ## --- トルク表示ココマデ ---

        # --- スラスター動作状況ココマデ ---

    def _draw_nav_data(self, player: RigidBody, target: RigidBody) -> None:
        """慣性計測装置の値および相対ナビゲーションを表示"""
        cx = 180
        cy = 100

        box_w = 200

        # --- 回転ココカラ ---

        offset = 10

        rotation_color = (100, 255, 100)
        radius = (box_w / 4) * 0.9

        ang_v_si = np.rad2deg(player.get_angular_velocity() / TU_TO_SEC)
        ang_v_si_ccw = max(0.0, ang_v_si)
        ang_v_si_cw = max(0.0, -ang_v_si)
        # 正の角速度（CCW）
        self._draw_circular_gauge(
            screen=self.screen,
            cx=cx - (box_w / 4),
            cy=cy - (box_w / 4) + offset,
            radius=radius,
            thickness=10,
            min_val=0.0,
            max_val=90,
            input_val=ang_v_si_ccw,
            full_color=rotation_color,
            center_labels=[f"{ang_v_si:.1f}", "°/s"],
            start_angle_rad=np.deg2rad(90),
            end_angle_rad=np.deg2rad(210),
            is_flipped_horizontally=False,
            is_gradation=False
        )
        # 負の角速度（CW）
        self._draw_circular_gauge(
            screen=self.screen,
            cx=cx - (box_w / 4),
            cy=cy - (box_w / 4) + offset,
            radius=radius,
            thickness=10,
            min_val=0.0,
            max_val=90,
            input_val=ang_v_si_cw,
            full_color=rotation_color,
            center_labels=[],
            start_angle_rad=np.deg2rad(90),
            end_angle_rad=np.deg2rad(210),
            is_flipped_horizontally=True,
            is_gradation=False
        )

        ang_acc_si = np.rad2deg(player.get_angular_acceleration() / (TU_TO_SEC ** 2))
        ang_acc_si_ccw = max(0.0, ang_acc_si)
        ang_acc_si_cw = max(0.0, -ang_acc_si)
        # 正の角加速度（CCW）
        self._draw_circular_gauge(
            screen=self.screen,
            cx=cx + (box_w / 4),
            cy=cy - (box_w / 4) + offset,
            radius=radius,
            thickness=10,
            min_val=0.0,
            max_val=3.0,
            input_val=ang_acc_si_ccw,
            full_color=(100, 255, 100),
            center_labels=[f"{ang_acc_si:.2f}", "°/s^2"],
            start_angle_rad=np.deg2rad(90),
            end_angle_rad=np.deg2rad(210),
            is_flipped_horizontally=False,
            is_gradation=False
        )
        # 負の角加速度（CW）
        self._draw_circular_gauge(
            screen=self.screen,
            cx=cx + (box_w / 4),
            cy=cy - (box_w / 4) + offset,
            radius=radius,
            thickness=10,
            min_val=0.0,
            max_val=3.0,
            input_val=ang_acc_si_cw,
            full_color=(100, 255, 100),
            center_labels=[],
            start_angle_rad=np.deg2rad(90),
            end_angle_rad=np.deg2rad(210),
            is_flipped_horizontally=True,
            is_gradation=False
        )

        # --- 回転ココマデ ---

        # --- 並進の正方形グリッド（相対速度・IMUの加速度）ココカラ ---

        max_value = 0.05 # 最大目盛り（速度と加速度で共通）
        interval_value = 0.01 # 目盛り間隔

        grid_w = box_w
        half_grid_w = grid_w // 2
        grid_interval = int(half_grid_w * (interval_value / max_value))

        grid_color = (100, 100, 100)

        grid_surf = pygame.Surface((grid_w, grid_w), pygame.SRCALPHA)

        # 外枠
        pygame.draw.circle(grid_surf, grid_color, (half_grid_w, half_grid_w), half_grid_w, 2)

        # XY軸
        pygame.draw.line(grid_surf, grid_color, (half_grid_w, 0), (half_grid_w, grid_w), 2)
        pygame.draw.line(grid_surf, grid_color, (0, half_grid_w), (grid_w, half_grid_w), 2)

        # ドッキング許容円
        allowable_radius = 0.01 # 許容半径
        allowable_radius_px = (allowable_radius / max_value) * half_grid_w
        pygame.draw.circle(grid_surf, (0, 255, 255), (half_grid_w, half_grid_w), allowable_radius_px, 1)

        # グリッド線
        for offset in range(grid_interval, half_grid_w + 1, grid_interval):
            theta = np.arccos(offset / half_grid_w)

            # 縦線
            for x in [half_grid_w - offset, half_grid_w + offset]:
                y0 = half_grid_w - (half_grid_w * np.sin(theta))
                y1 = half_grid_w + (half_grid_w * np.sin(theta))
                pygame.draw.line(grid_surf, grid_color, (x, y0), (x, y1), 1)

            # 横線
            for y in [half_grid_w - offset, half_grid_w + offset]:
                x0 = half_grid_w - (half_grid_w * np.sin(theta))
                x1 = half_grid_w + (half_grid_w * np.sin(theta))
                pygame.draw.line(grid_surf, grid_color, (x0, y), (x1, y), 1)

        grid_rect = grid_surf.get_rect(center=(cx, int(cy + (grid_w / 2))))
        self.screen.blit(grid_surf, grid_rect.topleft)

        ## --- ドット描画ココカラ ---

        orig_x_px = cx
        orig_y_px = cy + half_grid_w

        # プレイヤーの機体ローカル座標に回転変換するための値
        cos_t = np.cos(player.angle)
        sin_t = np.sin(player.angle)

        if player != target:
            # ターゲットのクリーナー衛星に対する相対速度ベクトル
            rel_v_world_si = (target.velocity - player.velocity) * (SEC_TO_TU / METER_TO_DU)
            
            rel_v_x = rel_v_world_si[0] * cos_t + rel_v_world_si[1] * sin_t
            rel_v_y = -rel_v_world_si[0] * sin_t + rel_v_world_si[1] * cos_t

            # 相対速度マーカーの位置
            rel_v_mag = np.hypot(rel_v_x, rel_v_y)
            if rel_v_mag > max_value:
                scale = max_value / rel_v_mag
                rel_v_x *= scale
                rel_v_y *= scale
            rel_v_y_px = orig_y_px - int((rel_v_x / max_value) * half_grid_w)
            rel_v_x_px = orig_x_px - int((rel_v_y / max_value) * half_grid_w)

            pygame.draw.line(self.screen, (255, 255, 0), (orig_x_px, orig_y_px), (rel_v_x_px, rel_v_y_px), 2)
            if rel_v_mag <= max_value:
                pygame.draw.circle(self.screen, (255, 255, 0), (rel_v_x_px, rel_v_y_px), 7)

        # クリーナー衛星のIMUが計測する加速度
        acc_imu = player.get_acc_from_imu()
        acc_x, acc_y = acc_imu[0], acc_imu[1]

        # 加速度マーカーの位置
        acc_mag = np.hypot(acc_x, acc_y)
        if acc_mag > max_value:
            scale = max_value / acc_mag
            acc_x *= scale
            acc_y *= scale
        acc_y_px = orig_y_px - int((acc_x / max_value) * half_grid_w)
        acc_x_px = orig_x_px - int((acc_y / max_value) * half_grid_w)

        pygame.draw.line(self.screen, (255, 0, 0), (orig_x_px, orig_y_px), (acc_x_px, acc_y_px), 2)
        if acc_mag <= max_value:
            pygame.draw.circle(self.screen, (255, 0, 0), (acc_x_px, acc_y_px), 5)

        ## --- ドット描画ココマデ ---

        for angle, sign in zip(np.arange(0, -np.pi * 2, -np.pi / 2), ['-', '', '', '-']):
            text_surf = self.font.render(f"{sign}{max_value:.2f} m/s", True, COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(
                cx + (half_grid_w * np.cos(angle)),
                cy + half_grid_w + (half_grid_w * np.sin(angle))
            ))
            self.screen.blit(text_surf, text_rect.topleft)
        
        text_surf = self.font.render("REL V", True, COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(
            cx - (half_grid_w * 0.9),
            cy + half_grid_w - (half_grid_w * 0.9)
        ))
        self.screen.blit(text_surf, text_rect.topleft)

        # --- 並進の正方形グリッド（相対速度・IMUの加速度）ココマデ ---

        return

    def _draw_bar_gauge(self, screen: pygame.Surface, cx: int, cy: int, w: int, h: int, angle: float, min_val: float,
                        max_val: float, input_val: float, full_color: tuple, stack_labels: list[str], is_gradation: bool):
        """
        ゲージコンポーネント

        Args:
            screen (pygame.Surface): 描画先
            cx (int): バーゲージ中心のx座標
            cy (int): バーゲージ中心のy座標
            w (int): 幅
            h (int): 高さ
            angle (float): バーゲージ中心まわりの回転角度（ラジアン）
            min_val (float): 最小値
            max_val (float): 最大値
            input_val (float): 入力値
            full_color (tuple): inputが最大の時の色
            stack_labels (list[str]): 縦積みで表示するテキスト
            is_gradation (bool): グラデーションフラグ
        """
        input_val = np.clip(input_val, min_val, max_val)
        normalized_input_val = (input_val - min_val) / (max_val - min_val)

        # グラデーションの場合，inputが0なら白，1ならfull_color．
        color = tuple(-((255 - ch) * normalized_input_val) + 255 for ch in full_color) if is_gradation else full_color

        # バーゲージ本体
        fill_h = int(h * normalized_input_val)
        gauge_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(gauge_surf, (50, 50, 50), (0, 0, w, h)) # 外枠
        pygame.draw.rect(gauge_surf, color, (0, h - fill_h, w, fill_h)) # 塗りつぶし
        rotated_surf = pygame.transform.rotate(gauge_surf, np.degrees(angle))
        gauge_rect = rotated_surf.get_rect(center=(cx, cy)) # 座標を指定してRectを生成
        screen.blit(rotated_surf, gauge_rect.topleft) # 転写

        # --- ラベル表示ココカラ ---

        if not stack_labels: return # ラベルが無ければ終了

        line_h = self.font.get_linesize() # フォント高さ
        spacing = 2

        label_surfs = [self.font.render(label, True, COLOR_UI_TEXT) for label in stack_labels] # 各ラベルに対するサーフェス

        # ラベル群全体のバウンディングボックス寸法
        total_text_w = max([surf.get_size()[0] for surf in label_surfs])
        total_text_h = sum([surf.get_size()[1] for surf in label_surfs]) + (spacing * (len(stack_labels) - 1))

        # 動的半径の計算と基準位置の決定
        theta = (np.pi / 2) + angle
        dir_x = np.cos(theta)
        dir_y = -np.sin(theta)

        # テキストボックスの指定方向への投影半径（出っ張り具合）を計算
        text_radius = (total_text_w / 2.0) * abs(dir_x) + (total_text_h / 2.0) * abs(dir_y)
        padding = 6 # テキストとゲージの間の余白（ピクセル）
        
        # ゲージ中心からの必要最低距離
        safe_radius = (h / 2.0) + text_radius + padding

        # 安全な基準位置（テキストブロックの中心点）
        base_x = gauge_rect.centerx + (safe_radius * dir_x)
        base_y = gauge_rect.centery + (safe_radius * dir_y)

        # 各ラベルの中心位置を計算して配置
        total_text_y_top = base_y - (total_text_h / 2)
        for i, label_surf in enumerate(label_surfs):
            label_center_y = total_text_y_top + ((2 * i) + 1) * (line_h / 2)
            label_rect = label_surf.get_rect(center=(base_x, label_center_y))
            screen.blit(label_surf, label_rect.topleft)

        # --- ラベル表示ココマデ ---

        return
    
    def _draw_time(self, mission_start_time: datetime, simulation_time: datetime, fast_forward_rate: float):
        """時間に関する描画"""
        # --- Mission Elapsed Timeココカラ ---

        total_sec = int((simulation_time - mission_start_time).total_seconds())

        sign = '+' if total_sec >= 0 else '-'
        total_sec = abs(total_sec)

        days = total_sec // 86400
        temp = total_sec % 86400

        hours = temp // 3600
        temp %= 3600

        minutes = temp // 60
        seconds = temp % 60

        met_str = f"MET{sign}{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
        met_surf = self.font.render(met_str, True, COLOR_UI_TEXT)
        met_rect = met_surf.get_rect(center=(self.screen.get_size()[0] // 2, 20))
        self.screen.blit(met_surf, met_rect.topleft)

        # --- Mission Elapsed Timeココマデ ---

        # 早送り倍率
        ff_rate_str = f"({fast_forward_rate:.0f}×)"
        ff_rate_surf = self.font.render(ff_rate_str, True, COLOR_UI_TEXT)
        ff_rate_rect = ff_rate_surf.get_rect(midleft=(met_rect.centerx + (met_surf.get_size()[0] // 2) + 10, 20))
        self.screen.blit(ff_rate_surf, ff_rate_rect.topleft)
    
    def _draw_fan_shape(self, screen: pygame.Surface, cx: int, cy: int, radius: int, thickness: int,
                        start_angle_rad: float, end_angle_rad: float, color: tuple, is_flipped_horizontally: bool):
        """
        扇形を描画するコンポーネント
        
        Args:
            screen (pygame.Surface): 描画対象のサーフェス
            cx, cy (int): 中心座標
            radius (int): 外径
            thickness (int): ゲージの太さ（ピクセル）
            start_angle_rad (float): 描画開始角度（ラジアン，画面右を0とし反時計回り）
            end_angle_rad (float): 描画終了角度（ラジアン）
            color (tuplr): 塗りつぶし色
            is_flipped_horizontally (bool): 左右反転フラグ
        """
        if start_angle_rad == end_angle_rad: # 角度差が0の場合
            return

        # 描画の滑らかさ（分割数）を角度の大きさに応じて動的に決定
        angle_diff = abs(end_angle_rad - start_angle_rad)
        steps = max(10, int(np.degrees(angle_diff) * 0.5)) 
        
        points = []
        sign = -1 if is_flipped_horizontally else 1
        
        # 外側の円弧の頂点を計算（開始角度 -> 終了角度）
        for i in range(steps + 1):
            theta = start_angle_rad + angle_diff * (i / steps)
            x = cx + sign * radius * np.cos(theta)
            y = cy - radius * np.sin(theta) # PyGameのY軸は下が正方向
            points.append((x, y))
            
        # 内側の円弧の頂点を計算（終了角度 -> 開始角度へと逆順で戻る）
        inner_radius = radius - thickness
        for i in range(steps, -1, -1):
            theta = start_angle_rad + angle_diff * (i / steps)
            x = cx + sign * inner_radius * np.cos(theta)
            y = cy - inner_radius * np.sin(theta) # PyGameのY軸は下が正方向
            points.append((x, y))
            
        # 頂点を結んで多角形として塗りつぶす．
        if len(points) > 2:
            pygame.draw.polygon(screen, color, points)
            pygame.draw.aalines(screen, color, True, points) # アンチエイリアスあり
    
    def _draw_circular_gauge(self, screen: pygame.Surface, cx: int, cy: int, radius: int, thickness: int,
                             min_val: float, max_val: float, input_val: float, full_color: tuple, center_labels: list[str],
                             start_angle_rad: float, end_angle_rad: float, is_flipped_horizontally: bool, is_gradation: bool):
        """
        扇形ゲージコンポーネント

        Args:
            screen (pygame.Surface): 描画先
            cx (int): 中心のx座標
            cy (int): 中心のy座標
            radius (int): 外径
            thickness (int): ゲージの太さ（ピクセル）
            min_val (float): 最小値
            max_val (float): 最大値
            input_val (float): 入力値
            full_color (tuple): inputが最大の時の色
            center_labels (list[str]): 中心に表示するテキスト
            start_angle_rad (float): 描画開始角度（ラジアン，画面右を0とし反時計回り）
            end_angle_rad (float): 描画終了角度（ラジアン）
            is_flipped_horizontally (bool): 左右反転フラグ
        """
        # 暗いグレーの背景
        self._draw_fan_shape(screen, cx, cy, radius, thickness, start_angle_rad, end_angle_rad, (50, 50, 50), is_flipped_horizontally)
        
        # --- メインの扇ココカラ ---

        input_val = np.clip(input_val, min_val, max_val)
        normalized_input_val = (input_val - min_val) / (max_val - min_val)

        # グラデーションの場合，inputが0なら白，1ならfull_color．
        color = tuple(-((255 - ch) * normalized_input_val) + 255 for ch in full_color) if is_gradation else full_color

        angle = normalized_input_val * (end_angle_rad - start_angle_rad)
        self._draw_fan_shape(screen, cx, cy, radius, thickness, start_angle_rad, start_angle_rad + angle, color, is_flipped_horizontally)

        # --- メインの扇ココマデ ---

        # --- ラベルココカラ ---

        if not center_labels: return

        label_surfs = [self.font.render(label, True, COLOR_UI_TEXT) for label in center_labels] # 各ラベルに対するサーフェス

        # ラベル群全体のバウンディングボックス寸法
        spacing = 2
        total_text_h = sum([surf.get_size()[1] for surf in label_surfs]) + (spacing * (len(center_labels) - 1))

        # 各ラベルの中心位置を計算して配置
        total_text_y_top = cy - (total_text_h / 2)
        line_h = self.font.get_linesize() # フォント高さ
        for i, label_surf in enumerate(label_surfs):
            label_center_y = total_text_y_top + ((2 * i) + 1) * (line_h / 2)
            label_rect = label_surf.get_rect(center=(cx, label_center_y))
            screen.blit(label_surf, label_rect.topleft)

        # --- ラベルココマデ ---
    
    def _setup_starry_sky(self):
        """起動時に1度だけ星をロード"""
        with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
        
        bright_df = df[df['magnitude'] <= 6.0]
        self.star_positions = Star.from_dataframe(bright_df)
        self.star_ra = np.radians(bright_df['ra_degrees'].values) # 星の赤経（RA）
        self.star_dec = np.radians(bright_df['dec_degrees'].values) # 星の赤緯（DEC）
        self.star_mags = bright_df['magnitude'].values

        self.ts = load.timescale()
        self.star_color_tint = (0.7, 0.85, 1.0) # 星空の色温度（Tint）設定
    
    def draw_starry_sky(self, simulation_time: datetime):
        """背景の星空を描画する"""
        t = self.ts.from_datetime(simulation_time)
        gast_hours = t.gast # グリニッジ視恒星時（0〜24時）
        
        # メキシコ上空（GAST - 6h）から地球を見る時，視線の向きはさらに+12時間反対側になる．
        center_ra_hours = (gast_hours - 6.0 + 12.0) % 24.0 # 地方恒星時
        center_ra_rad = center_ra_hours * (np.pi / 12.0) # 2piラジアン / 24時間

        half_w = self.screen.get_width() // 2
        half_h = self.screen.get_height() // 2

        # 画面の対角線距離（ピクセル）を計算
        # 画面がどう回転しても，中心から画面の端までの最大距離はコレになる．
        diagonal = np.hypot(half_w, half_h)

        # カメラの倍率に依存しない固定スケール（カメラの視野角は固定で，ドリーイン・ドリーアウトする．）
        # 宇宙の最も「狭い」方向は赤緯（DEC）の ±pi/2 ラジアン．この ±pi/2 が，画面の対角線をすっぽり覆うようにスケール．
        fixed_scale = diagonal / (np.pi / 2.0)

        # 赤経の差分を -pi から pi に収める．
        d_ra = (self.star_ra - center_ra_rad + np.pi) % (2 * np.pi) - np.pi
        
        x_coords = half_w - (d_ra * fixed_scale) # 天球の内側から見て東（RA大）は左側なのでマイナス
        y_coords = half_h - (self.star_dec * fixed_scale) # 北極（DEC大）は画面上なのでマイナス

        if isinstance(self.camera, RelativeCamera):
            # ターゲットの「地球に対する角度（位相）」を計算
            target_pos = self.camera.get_target_body().position
            theta = np.atan2(target_pos[1], target_pos[0])
            
            rot_angle = (np.pi / 2.0) - theta # ターゲットを真上に持ってくるために回している角度
            
            # 画面中心を原点とした相対座標（星は遠すぎるため見え方が不変で，画面中心の回転でよい．）
            dx = x_coords - half_w
            dy = y_coords - half_h
            
            cos_t = np.cos(rot_angle)
            sin_t = np.sin(rot_angle)
            
            # Y軸下向き座標系での回転演算
            rotated_x = dx * cos_t + dy * sin_t
            rotated_y = -dx * sin_t + dy * cos_t
            
            # 再び画面座標に戻す．
            x_coords = rotated_x + half_w
            y_coords = rotated_y + half_h
        
        # 画面内に収まっている星のインデックスを取得（カリング）
        visible_mask = (x_coords > -10) & (x_coords < (half_w * 2) + 10) & \
                       (y_coords > -10) & (y_coords < (half_h * 2) + 10)
        
        visible_x = x_coords[visible_mask]
        visible_y = y_coords[visible_mask]
        visible_mags = self.star_mags[visible_mask]

        # 描画
        tint_r, tint_g, tint_b = self.star_color_tint
        for x, y, mag in zip(visible_x, visible_y, visible_mags):
            # 等級が小さいほど明るく，大きな円で描く．
            size = max(1, int((6.0 - mag) * 0.7))
            
            # ベースとなる輝度（グレースケール）
            brightness = max(50, min(255, int(255 - (mag - 1.0) * 40)))
            
            color = (
                int(brightness * tint_r), # 赤成分を減らす
                int(brightness * tint_g), # 緑成分を少し減らす
                int(brightness * tint_b)  # 青成分はそのまま（最大値）
            )
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
    
    def _draw_capture_ui(self, capture_state: str, progress: float):
        """捕獲状態とプログレスバーの描画"""
        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()

        # 状態テキストの描画
        state_color = COLOR_UI_TEXT
        text = "ARM: IDLE (APPROACH TARGET)"
        if capture_state == "IDLE":
            state_color = (255, 200, 0)
            text = "ARM: IDLE (APPROACH TARGET)"
        elif capture_state == "CAPTURING":
            state_color = (0, 255, 255)
            text = "CAPTURING... KEEP POSITION!"
        elif capture_state == "DOCKED":
            state_color = (0, 255, 0)
            text = "TARGET DOCKED! (PRESS ENTER TO RELEASE)"
        
        text_surface = self.font.render(text, True, state_color)
        self.screen.blit(text_surface, (screen_w // 2 - text_surface.get_width() // 2, screen_h - 80))

        # プログレスバーの描画
        self._draw_bar_gauge(
            screen=self.screen,
            cx=screen_w // 2,
            cy=screen_h - 40,
            w=20,
            h=350,
            angle=-np.pi / 2,
            min_val=0.0,
            max_val=1.0,
            input_val=progress,
            full_color=state_color,
            stack_labels=[f"{int(progress * 100)}%"],
            is_gradation=False
        )
    
    def _draw_fuel_gage(self, player: RigidBody):
        """燃料ゲージの描画"""
        screen_w = self.screen.get_width()

        offset = 100
        radius = 70

        self._draw_circular_gauge(
            screen=self.screen,
            cx=screen_w - offset,
            cy=offset,
            radius=radius,
            thickness=10,
            min_val=0.0,
            max_val=player.max_propellant_mass,
            input_val=player.propellant_mass,
            full_color=(100, 150, 255),
            center_labels=[f"{player.propellant_mass / KG_TO_MU:.1f} kg", 'Fuel'],
            start_angle_rad=np.deg2rad(-30),
            end_angle_rad=np.deg2rad(210),
            is_flipped_horizontally=True,
            is_gradation=False
        )

        return
