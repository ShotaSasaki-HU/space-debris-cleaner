# view/renderer.py
import pygame
import numpy as np
from typing import Dict
from datetime import datetime

from physics.body import RigidBody
from view.camera import Camera
from physics.constants import METER_TO_DU, SEC_TO_TU, MAX_THRUST_NEWTON, MAX_TORQUE_NM, NM_TO_CANONICAL

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

    def draw_bodies(self, bodies: list[RigidBody], selected_body: RigidBody):
        """宇宙の天体・オブジェクトを描画する"""
        for body in bodies:
            if body.is_fixed:
                earth_pos = self.camera.world_to_screen(body.position)
                earth_r = self.camera.pixels_per_du
                
                screen_w, screen_h = self.screen.get_size()
                
                # 地球のバウンディングボックス（外接矩形）が画面の範囲内にあるか判定
                is_on_screen = not (
                    earth_pos[0] + earth_r < 0 or 
                    earth_pos[0] - earth_r > screen_w or
                    earth_pos[1] + earth_r < 0 or 
                    earth_pos[1] - earth_r > screen_h
                )
                
                if is_on_screen:
                    # Pygameのdraw.circleは巨大すぎる半径を与えるとフリーズするため，安全な上限を設ける．
                    if earth_r < 30000:
                        self._draw_realistic_body(body=body, is_selecetd=False)
                    else:
                        # 描画限界を超えるズームで地球に肉薄している（または中にいる）場合，パフォーマンス保護のため画面全体を地球色で塗りつぶす．
                        self.screen.fill(COLOR_EARTH)
            else:
                is_selected = (body is selected_body)
                self._draw_realistic_body(body=body, is_selecetd=is_selected)

    def _draw_realistic_body(self, body: RigidBody, is_selecetd: bool = False):
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
                # スケーリングロジックの決定
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

                scaled_image = pygame.transform.smoothscale(image, (target_w, target_h))

                # Cameraの場合
                if isinstance(self.camera, Camera):
                    rotated_image = pygame.transform.rotate(scaled_image, np.rad2deg(body.angle))
                # RelativeCameraの場合
                else:
                    # ターゲットの「地球に対する角度（位相）」を計算
                    theta = np.atan2(self.camera.target_body.position[1], self.camera.target_body.position[0])
                    rotated_image = pygame.transform.rotate(scaled_image, np.rad2deg((np.pi / 2) - theta + body.angle))
                
                screen_pos = self.camera.world_to_screen(body.position) # 物理エンジンの座標を画面座標へ変換
                rotated_rect = rotated_image.get_rect() # 回転後の画像サイズを取得
                draw_pos = (screen_pos[0] - rotated_rect.width // 2,
                            screen_pos[1] - rotated_rect.height // 2) # 画像の中心座標を取得
                
                # 選択されているbodyの縁取り
                if is_selecetd:
                    opaque_mask = pygame.mask.from_surface(rotated_image) # 画像の非透過部分からマスクを生成
                    outline = opaque_mask.outline() # マスクの輪郭座標リスト
                    if len(outline) >= 2:
                        # 輪郭座標は画像の左上が原点なので，画面描画座標 (draw_pos) にオフセットを加算する．
                        outline_points = [(p[0] + draw_pos[0], p[1] + draw_pos[1]) for p in outline]

                        # 縁取りの透過
                        temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                        pygame.draw.lines(temp_surf, (255, 255, 0, 255), True, outline_points, 2)
                        self.screen.blit(temp_surf, (0, 0))

                self.screen.blit(rotated_image, draw_pos)
                return # リアル画像での描画に成功したら終了

        # フォールバック（画像がない場合の描画）
        body_screen_pos = self.camera.world_to_screen(body.position)
        pygame.draw.rect(self.screen, (255, 255, 255), (body_screen_pos[0]-6, body_screen_pos[1]-6, 12, 12))

    def draw_ui(self, player: RigidBody, target: RigidBody, sas_enabled: bool, throttle: float,
                player_torque: float, mission_start_time: datetime, simulation_time: datetime, fast_forward_rate: float):
        """各種UIを描画する"""
        self._draw_rel_nav_ui(player, target)
        self._draw_control_console(sas_enabled, throttle, player, player_torque)
        self._draw_time(mission_start_time, simulation_time, fast_forward_rate)

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

    def _draw_control_console(self, sas_enabled: bool, throttle: float, player: RigidBody, player_torque: float):
        """操作に関するテキスト表示"""
        # UIの一番上に現在のカメラモードを描画
        mode_text = "VIEW: " + ("EARTH" if isinstance(self.camera, Camera) else "TRACKING")
        self.screen.blit(self.font.render(mode_text, True, COLOR_UI_TEXT), (20, 20))
        
        sas_text = "SAS: ON" if sas_enabled else "SAS: OFF"
        sas_color = (100, 255, 100) if sas_enabled else (200, 200, 200)
        self.screen.blit(self.font.render(sas_text, True, sas_color), (20, 40))
        
        help_color = (150, 150, 150)
        self.screen.blit(self.font.render("W/S: Forward/Backward | A/D: Left/Right", True, help_color), (20, 60))
        self.screen.blit(self.font.render("Q/E: Manual Rotation (SAS OFF) | R: Toggle SAS", True, help_color), (20, 80))

        # スロットル
        self._draw_bar_gauge(screen=self.screen, cx=self.screen.get_width() - 200, cy=self.screen.get_height() - 200, w=50, h=150,
                             angle=0.0, min_val=0.0, max_val=1.0, input_val=throttle, full_color=(255, 0, 0),
                             stack_labels=['Throttle', f'{throttle * 100:.0f}%'], is_gradation=False)

        # --- スラスター動作状況ココカラ ---

        cx = 180
        cy = self.screen.get_size()[1] // 2
        
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
            pygame.draw.polygon(self.screen, COLOR_PLAYER, [
                (cx, cy - 30), (cx - 20, cy + 20), (cx + 20, cy + 20)
            ])

        keys = pygame.key.get_pressed()
        
        # 4方向のバーゲージの描画
        bar_w = 20
        bar_h = 50
        offset = 60 # 画像中心からの距離

        # 噴射時はスロットル値に比例，非噴射時は0．
        val_w = throttle if keys[pygame.K_w] else 0.0
        val_s = throttle if keys[pygame.K_s] else 0.0
        val_a = throttle if keys[pygame.K_a] else 0.0
        val_d = throttle if keys[pygame.K_d] else 0.0

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
            labels=[],
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
            labels=[],
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
                             min_val: float, max_val: float, input_val: float, full_color: tuple, labels: list[str],
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
            labels (list[str]): 中心に表示するテキスト
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

        if not labels: return

        label_surfs = [self.font.render(label, True, COLOR_UI_TEXT) for label in labels] # 各ラベルに対するサーフェス

        # ラベル群全体のバウンディングボックス寸法
        spacing = 2
        total_text_h = sum([surf.get_size()[1] for surf in label_surfs]) + (spacing * (len(labels) - 1))

        # 各ラベルの中心位置を計算して配置
        total_text_y_top = cy - (total_text_h / 2)
        line_h = self.font.get_linesize() # フォント高さ
        for i, label_surf in enumerate(label_surfs):
            label_center_y = total_text_y_top + ((2 * i) + 1) * (line_h / 2)
            label_rect = label_surf.get_rect(center=(cx, label_center_y))
            screen.blit(label_surf, label_rect.topleft)

        # --- ラベルココマデ ---
