# main.py
import pygame
import sys
import numpy as np
from datetime import datetime, timedelta, timezone
from enum import Enum, auto

from physics.engine import GravityEngine
from physics.body import RigidBody
from physics.constants import (
    KG_TO_MU, EARTH_MASS_KG, METER_TO_DU, EARTH_RADIUS_M, G_CANONICAL, TU_TO_SEC, SEC_TO_TU,
    CLEANER_SAT_MASS_KG, CLEANER_SAT_MOMENT_OF_INERTIA_KG_M2, CLEANER_SAT_SIZE_METER,
    MAX_THRUST_NEWTON, MAX_TORQUE_NM, NEWTON_TO_CANONICAL, NM_TO_CANONICAL, ATMOSPHERE_RADIUS_DU
)
from physics.control import PIDController
from view.camera import EarthCamera, RelativeCamera
from view.renderer import GameRenderer
from utils.loader import LevelLoader
from utils.audio import ThrusterAudioManager

# アプリケーション全体の設定
SCREEN_WIDTH_INIT = 1280
SCREEN_HEIGHT_INIT = 720
FPS = 60
PIXELS_PER_DU = 200.0
TIME_STEP_TU_PHYSICS = (1 / FPS) * SEC_TO_TU # 物理エンジンの微小ステップ幅

class GameState(Enum):
    TITLE = auto()    # タイトル画面
    PLAYING = auto()  # ゲーム本編
    CLEAR = auto()    # ミッション成功
    GAMEOVER = auto() # ミッション失敗

class SpaceDebrisApp:
    """
    ゲームアプリケーションのメインクラス．
    初期化，イベント処理，メインループの制御を行う．
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH_INIT, SCREEN_HEIGHT_INIT), pygame.RESIZABLE)
        pygame.display.set_caption("Space Debris Cleaner")
        self.clock = pygame.time.Clock()
        self.running = True

        self.thruster_audio = ThrusterAudioManager(
            loop_wav_path="assets/sounds/RcsHeavy.wav",
            shutoff_wav_path="assets/sounds/RcsHeavyShutoff.wav"
        )

        self.capture_time_required_sec = 10.0 # 捕獲完了に必要な継続接触時間（秒）

        self._reset_game() # 動的なステートの初期化
        self.state = GameState.TITLE # 初期状態はタイトル画面
    
    def _reset_game(self):
        self.throttle = 1.0 # スロットル100%
        self.player_torque = 0.0
        self.fast_forward_rate = 1.0 # 早送り倍率
        self.time_accumulator = 0.0 # 未処理のシミュレーション時間を貯めるバケツ
        self.simulation_time = datetime.now(timezone.utc) # ゲーム内の時刻（物理演算には関係しない．）
        self.mission_start_time = self.simulation_time
        self.orbital_predictions: dict = {}

        # 捕獲システムのステートマシンと変数
        self.capture_state = 'IDLE' # 'IDLE', 'CAPTURING', 'DOCKED'
        self.capture_progress = 0.0 # 捕獲の進捗（0.0〜1.0）

        self.state = GameState.PLAYING # リセット処理（Rキー）が呼ばれたら，即座にプレイ画面から再開する．
        self.end_reason = "" # ゲーム終了理由のテキスト
        self.is_cinematic_mode = False # 燃え尽きるまで見届けるモードか否か
        self.doomed_debri = None # 再突入が確定したデブリのインスタンス
        self.deorbited_debris_count = 0 # 消滅済みのデブリ数（実績）

        # 各モジュールを新しいインスタンスで上書きして初期化
        self._setup_physics()
        self._setup_view()
        self._setup_controls()

    def _setup_physics(self):
        """物理エンジンと天体の初期配置"""
        self.engine = GravityEngine(
            time_step=TIME_STEP_TU_PHYSICS,
            surface_radius_du=EARTH_RADIUS_M * METER_TO_DU,
            atmosphere_radius_du=ATMOSPHERE_RADIUS_DU
        )

        # 地球
        M_earth = KG_TO_MU * EARTH_MASS_KG
        self.earth = RigidBody(
            mass=M_earth,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            is_fixed=True,
            image_path="assets/images/earth.png",
            real_width_du=2,
            real_height_du=2,
            draw_fixed_size_px=30
        )
        self.engine.add_body(self.earth)

        debris_list = LevelLoader.load_debris_from_json("assets/debris_config.json")
        for debri in debris_list:
            self.engine.add_body(debri)
        self.selected_body = self.engine.bodies[-1] # 初期ターゲットの設定（プレイヤー追加前に実施）

        # プレイヤー
        r_player = METER_TO_DU * (EARTH_RADIUS_M + 400e3)
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
            draw_fixed_size_px=30,
            isp_sec=220.0
        )
        self.engine.add_body(self.player_sat)

        self.engine.initialize()

        self.max_thrust_cano = MAX_THRUST_NEWTON * NEWTON_TO_CANONICAL
        self.max_torque_cano = MAX_TORQUE_NM * NM_TO_CANONICAL

    def _setup_view(self):
        """描画関連の初期化"""
        self.earth_camera = EarthCamera(self.screen, PIXELS_PER_DU)

        self.tracking_camera = RelativeCamera(self.screen, PIXELS_PER_DU)
        self.tracking_camera.set_target_body(self.selected_body)

        self.view_mode = "EARTH"
        self.renderer = GameRenderer(self.screen, self.earth_camera)

    def _setup_controls(self):
        """入力制御系の初期化"""
        self.sas_enabled = False
        self.sas_controller = PIDController(kp=0.5, ki=0.0, kd=5.0)
    
    def _handle_playing_events(self, event):
        """ゲーム本編プレイ中の入力処理"""

        # --- マウスクリックによるターゲット選択ココカラ ---

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 左クリック
                mouse_x, mouse_y = event.pos

                for body in reversed(self.engine.bodies): # 後から描画された方を優先
                    if body.is_fixed: continue

                    screen_x, screen_y = self.renderer.camera.world_to_screen(body.position) # オブジェクトの画面上のピクセル座標
                    dist = np.hypot(screen_x - mouse_x, screen_y - mouse_y)

                    target_w_px = int(body.real_width_du * self.renderer.camera.pixels_per_du)
                    target_h_px = int(body.real_height_du * self.renderer.camera.pixels_per_du)
                    if min(target_w_px, target_h_px) < body.draw_fixed_size_px:
                        target_r_px = body.draw_fixed_size_px // 2
                    else:
                        target_r_px = max(target_w_px, target_h_px) // 2

                    if dist < target_r_px:
                        self.selected_body = body
                        self.tracking_camera.set_target_body(body)
                        break # 先に見つかったbodyが優先

        # --- マウスクリックによるターゲット選択ココマデ ---

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.sas_enabled = not self.sas_enabled

            # カメラの3段階切り替えロジック
            elif event.key == pygame.K_RSHIFT:
                if self.view_mode == "EARTH":
                    self.view_mode = "TRACKING"
                    self.renderer.camera = self.tracking_camera
                else:
                    self.view_mode = "EARTH"
                    self.renderer.camera = self.earth_camera

            # 捕獲・リリース操作
            elif event.key == pygame.K_RETURN:
                if self.capture_state in ['CAPTURING', 'DOCKED']:
                    if self.capture_state == 'DOCKED':
                        # リリースして，物理エンジンに再び単体の剛体として戻す．
                        released_body = self.player_sat.undock()
                        self.engine.add_body(released_body)
                        
                    self.capture_state = 'IDLE'
                    self.capture_progress = 0.0
                
            if type(self.renderer.camera) is EarthCamera:
                if event.key == pygame.K_RIGHT:
                    max_pixels_per_du = min(self.renderer.camera.screen_width, self.renderer.camera.screen_height) / (1.3 * 2.0) # 地球の直径 = 2DU
                    self.renderer.camera.set_pixels_per_du(min(max_pixels_per_du, self.renderer.camera.get_pixels_per_du() * 2))
                elif event.key == pygame.K_LEFT:
                    self.renderer.camera.set_pixels_per_du(max(30, self.renderer.camera.get_pixels_per_du() // 2))
            elif type(self.renderer.camera) is RelativeCamera:
                if event.key == pygame.K_RIGHT:
                    target_body = self.renderer.camera.get_target_body()
                    required_du = 1.2 * np.linalg.norm([target_body.real_width_du, target_body.real_height_du])
                    max_pixels_per_du = min(self.renderer.camera.screen_width, self.renderer.camera.screen_height) / required_du
                    self.renderer.camera.set_pixels_per_du(min(max_pixels_per_du, self.renderer.camera.get_pixels_per_du() * 2))
                elif event.key == pygame.K_LEFT:
                    self.renderer.camera.set_pixels_per_du(max(PIXELS_PER_DU, self.renderer.camera.get_pixels_per_du() // 2))

    def _handle_common_hotkeys(self, event):
        """プレイ中・結果確定後を問わず有効な共通ホットキー"""
        if event.type == pygame.KEYDOWN:
            # 早送り係数の操作
            if event.key == pygame.K_PERIOD:
                self.fast_forward_rate = min(1000.0, self.fast_forward_rate * 10.0)
            elif event.key == pygame.K_COMMA:
                self.fast_forward_rate = max(1.0, self.fast_forward_rate / 10.0)

    def handle_events(self):
        """ユーザー入力の処理"""
        for event in pygame.event.get():

            # --- システムレベルのイベント（全ステート共通）ココカラ ---

            if event.type == pygame.QUIT:
                self.running = False
            
            # ウィンドウリサイズ
            elif event.type == pygame.VIDEORESIZE:                
                # 新しいサイズのSurfaceを再生成
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
                # 古いSurfaceの参照を，新しいSurfaceで上書きする．
                if hasattr(self, 'renderer'):
                    self.renderer.screen = self.screen
                if hasattr(self, 'earth_camera'):
                    self.earth_camera.update_screen_size(self.screen)
                if hasattr(self, 'tracking_camera'):
                    self.tracking_camera.update_screen_size(self.screen)
            
            # --- システムレベルのイベント（全ステート共通）ココマデ ---
            
            # --- 状態ごとの入力処理ココカラ ---

            if self.state == GameState.TITLE:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: # ゲーム開始
                    self.state = GameState.PLAYING

            elif self.state == GameState.PLAYING:
                self._handle_playing_events(event)
                self._handle_common_hotkeys(event)
            
            elif self.state in (GameState.CLEAR, GameState.GAMEOVER):
                self._handle_common_hotkeys(event) # 結果確定後も共通ホットキーを評価
                
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self._reset_game()
            
            # --- 状態ごとの入力処理ココマデ ---
        
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
        total_thrust = 0.0 # 消費した推力の絶対値の合計
        if keys[pygame.K_w]: 
            thrust_x += thrust_mag
            total_thrust += thrust_mag
        if keys[pygame.K_s]: 
            thrust_x -= thrust_mag
            total_thrust += thrust_mag
        if keys[pygame.K_a]: 
            thrust_y += thrust_mag
            total_thrust += thrust_mag
        if keys[pygame.K_d]: 
            thrust_y -= thrust_mag
            total_thrust += thrust_mag

        # オーディオ更新
        has_fuel = self.player_sat.propellant_mass > 0
        self.thruster_audio.update_thruster(thruster_id="K_w", is_firing=keys[pygame.K_w] and has_fuel)
        self.thruster_audio.update_thruster(thruster_id="K_s", is_firing=keys[pygame.K_s] and has_fuel)
        self.thruster_audio.update_thruster(thruster_id="K_a", is_firing=keys[pygame.K_a] and has_fuel)
        self.thruster_audio.update_thruster(thruster_id="K_d", is_firing=keys[pygame.K_d] and has_fuel)

        if total_thrust > 0: # 合力が0でも，スラスターが動作していれば処理を呼ぶ．
            # DOCKED状態（重心がズレている）なら，オフセット位置から推力を加える．
            if hasattr(self.player_sat, 'visual_offset_local') and np.any(self.player_sat.visual_offset_local):
                ox, oy = self.player_sat.visual_offset_local
                self.player_sat.apply_local_force_at_offset(thrust_x, thrust_y, ox, oy, total_thrust, dt_tu)
            else:
                self.player_sat.apply_local_force(thrust_x, thrust_y, total_thrust, dt_tu)
        
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
    
    def _check_capture_contact(self) -> bool:
        """アーム先端がターゲットデブリに接触しているか判定"""
        if not self.selected_body: return False

        # アーム先端の物理座標を計算
        arm_length_du = self.player_sat.real_width_du / 2.0
        angle = self.player_sat.angle
        tip_world_pos = self.player_sat.position + np.array([
            arm_length_du * np.cos(angle),
            arm_length_du * np.sin(angle)
        ])

        # ターゲットのpositionからアーム先端への相対位置ベクトル
        rel_pos = tip_world_pos - self.selected_body.position
        target_angle = self.selected_body.angle

        # ターゲットの回転を打ち消す方向に，相対位置ベクトルを回す．
        cos_t = np.cos(-target_angle)
        sin_t = np.sin(-target_angle)
        local_x_du = rel_pos[0] * cos_t - rel_pos[1] * sin_t
        local_y_du = rel_pos[0] * sin_t + rel_pos[1] * cos_t

        # Rendererのキャッシュからデブリの画像を取得
        image = self.renderer.image_cache.get(self.selected_body.image_path)
        if not image: return False

        debri_w_px, debri_h_px = image.get_size()

        # 水平なターゲットのpositionからアーム先端への相対位置ベクトル（px）
        local_x_px = (local_x_du / self.selected_body.real_width_du) * debri_w_px
        local_y_px = (-local_y_du / self.selected_body.real_height_du) * debri_h_px

        # ターゲット画像の左上(0, 0)からアーム先端への位置ベクトルへ変換
        local_x_px = int(local_x_px + (debri_w_px / 2))
        local_y_px = int(local_y_px + (debri_h_px / 2))

        mask = pygame.mask.from_surface(image, threshold=127)
        # アームの先端が「デブリ画像の範囲内」かつ「マスクが不透明」なら接触
        if (0 <= local_x_px < debri_w_px) and (0 <= local_y_px < debri_h_px):
            return mask.get_at((local_x_px, local_y_px)) != 0

        return False
    
    def _update_playing(self, dt_real_sec: float):
        """
        ゲーム本編プレイ中の更新処理を行う．

        Args:
            dt_real_sec (float): 前フレームから経過した現実の時間（秒）
        """
        self.time_accumulator += dt_real_sec * SEC_TO_TU * self.fast_forward_rate

        if self.fast_forward_rate >= 1000:
            current_physics_dt_tu = TIME_STEP_TU_PHYSICS * 10**(int(np.log10(self.fast_forward_rate)) - 2)
        else:
            current_physics_dt_tu = TIME_STEP_TU_PHYSICS
        
        self.engine.set_time_step(current_physics_dt_tu) # エンジン内部の時間幅を動的に書き換える．

        while self.time_accumulator >= current_physics_dt_tu:
            self._apply_control_forces(dt_tu=current_physics_dt_tu) # ユーザによる並進・回転入力の評価

            events = self.engine.step()
            for event in events:
                if event.body1_destroyed and (event.body1 in self.engine.bodies):
                    self.engine.remove_body(event.body1)
                    if (event.body1 not in (self.player_sat, self.earth)) and (event.body2 == self.earth):
                        self.deorbited_debris_count += 1

                if event.body2_destroyed and (event.body2 in self.engine.bodies):
                    self.engine.remove_body(event.body2)
                    if (event.body2 not in (self.player_sat, self.earth)) and (event.body1 == self.earth):
                        self.deorbited_debris_count += 1
            
            # --- 捕獲判定ココカラ ---

            if self.capture_state != 'DOCKED':
                if self.fast_forward_rate > 5.0: # 早送り中は捕獲判定をスキップ
                    is_touching = False
                    is_slow_enough = False
                else:
                    is_touching = self._check_capture_contact()

                    # 相対速度の閾値チェック
                    v_rel = np.linalg.norm(self.player_sat.velocity - self.selected_body.velocity)
                    v_rel_si = v_rel * (SEC_TO_TU / METER_TO_DU)
                    is_slow_enough = v_rel_si <= 0.01 # 0.01m/s以下 <=> 1cm/s以下

                if is_touching and is_slow_enough:
                    self.capture_state = 'CAPTURING'
                    # プログレスを進める．（1ステップ分の時間を加算）
                    current_physics_dt_sec = current_physics_dt_tu * TU_TO_SEC
                    self.capture_progress += current_physics_dt_sec / self.capture_time_required_sec

                    if self.capture_progress >= 1.0:
                        self.capture_state = 'DOCKED'
                        self.capture_progress = 1.0
                        
                        # エンジンからデブリを削除し，プレイヤーに結合する．
                        self.engine.remove_body(self.selected_body)
                        self.player_sat.dock_with(self.selected_body)

                        # 追跡対象切り替え
                        self.selected_body = self.player_sat
                        self.tracking_camera.set_target_body(self.selected_body)
                else:
                    # 離れたり早すぎたりしたら捕獲プログレスをリセット
                    if self.capture_state == 'CAPTURING':
                        self.capture_state = 'IDLE'
                        self.capture_progress = 0.0

            # --- 捕獲判定ココマデ ---

            self.time_accumulator -= current_physics_dt_tu
            self.simulation_time += timedelta(seconds=current_physics_dt_tu * TU_TO_SEC) # ループの外でもほぼ問題ないが，厳密を期すならココ．

        # 軌道予測は重いため，1フレームに1回だけ実行する．
        self.orbital_predictions = self.engine.predict_trajectories(future_duration=30.0, dt_prediction=0.05)

    def update(self, dt_real_sec: float):
        """
        ゲーム状態に応じた更新処理を行う．

        Args:
            dt_real_sec (float): 前フレームから経過した現実の時間（秒）
        """
        if self.state == GameState.TITLE:
            pass
        elif self.state == GameState.PLAYING:
            self._update_playing(dt_real_sec) # ゲーム本編プレイ中の更新処理
            self._check_win_loss_condition() # 勝敗判定
        elif self.state in (GameState.CLEAR, GameState.GAMEOVER):
            self._update_playing(dt_real_sec) # ゲーム本編プレイ中の更新処理を流用
    
    def _check_win_loss_condition(self):
        """ゲームの終了条件を監視"""
        if self.state != GameState.PLAYING: return

        player_doomed = False # doomed: 消える運命にある
        debris_doomed = (self.deorbited_debris_count > 0) # 既に落としたデブリがあれば，その時点でデブリ側の条件はクリアとする．

        # 予測軌道から「大気圏に突入する運命にあるか」を判定
        for body_id, path in self.orbital_predictions.items():
            if not path: continue

            # 地球は判定から除外（含んでいると無条件でdebris_doomedが立ってしまう．）
            if body_id == id(self.earth): continue

            # 軌道のうち最も地球に近い点の距離
            min_r = min([np.linalg.norm(p) for p in path])

            # 大気圏の境界を下回る未来が確定しているならドゥームフラグを立てる．
            if min_r < ATMOSPHERE_RADIUS_DU:
                if body_id == id(self.player_sat):
                    player_doomed = True
                else:
                    debris_doomed = True
                    self.doomed_debri = next((b for b in self.engine.bodies if id(b) == body_id), None)
        
        # ドッキング中なら，プレイヤーの運命＝デブリの運命として扱う．
        if (self.capture_state == 'DOCKED') and player_doomed:
            debris_doomed = True
        
        # 予測ではなく，現在位置が実際に地表に激突しているか．
        # または，物理エンジンの管理下から既に削除されているか．
        is_player_crashed = (
            np.linalg.norm(self.player_sat.position) <= (EARTH_RADIUS_M * METER_TO_DU) or
            self.player_sat not in self.engine.bodies
        )

        previous_state = self.state # 状態遷移を検知して早送り倍率を設定するための記録
        
        # --- 成否のステートマシンココカラ ---

        # 成功条件：デブリとプレイヤー双方が突入見込み（予測線で確定）
        if debris_doomed and player_doomed:
            self.state = GameState.CLEAR
            self.end_reason = "Re-entry trajectory confirmed!"
            self.is_cinematic_mode = True # 燃え尽きるまで見せるモードON
        
        # 失敗条件：プレイヤーが地表に激突して物理的に破壊された（燃料が残っていても死）
        # 成功条件をすり抜けているため，後からデブリが大気圏突入する可能性も無い．
        elif is_player_crashed:
            self.state = GameState.GAMEOVER

            # 高度で死因を切り分ける
            if np.linalg.norm(self.player_sat.position) <= (EARTH_RADIUS_M * METER_TO_DU):
                self.end_reason = "Hull destroyed on impact with Earth's surface."
            else:
                self.end_reason = "Hull destroyed due to critical collision."

            self.is_cinematic_mode = False
        
        # 失敗条件
        elif (not debris_doomed) and ((self.player_sat.propellant_mass <= 0.0) and player_doomed):
            self.state = GameState.GAMEOVER
            self.end_reason = "Fatal trajectory. Target not removed."
            self.is_cinematic_mode = True # 燃え尽きるまで見せるモードON
        
        # 失敗条件：燃料がゼロになった時点で，CLEAR条件を満たしていないなら全て失敗．
        elif self.player_sat.propellant_mass <= 0.0:
            self.state = GameState.GAMEOVER
            self.end_reason = "Out of propellant. Stranded in orbit."
            self.is_cinematic_mode = False # 永遠に軌道を回り続けるので即座にリザルトへ

        # --- 成否のステートマシンココマデ ---

        # ステートが切り替わった「その瞬間」に1度だけ共通の終了処理を行う．
        if previous_state == GameState.PLAYING and self.state != GameState.PLAYING:
            # シネマティックモードの場合，強制的にトラッキングカメラにして最期を見せる．
            if self.is_cinematic_mode:
                self.renderer.camera = self.tracking_camera

            self.fast_forward_rate = 10.0

    def render(self):
        """画面の描画"""
        self.renderer.clear()
        self.renderer.draw_starry_sky(simulation_time=self.simulation_time)
        self.renderer.draw_predictions(self.orbital_predictions, player=self.player_sat, selected_body=self.selected_body)
        self.renderer.draw_bodies(bodies=self.engine.bodies, selected_body=self.selected_body)
        
        # ゲーム状態に応じたUIを上塗り
        if self.state == GameState.TITLE:
            self.renderer.draw_overlay("SPACE DEBRIS CLEANER", "Press 'SPACE' to start.", (50, 150, 255))
        elif self.state == GameState.PLAYING:
            self.renderer.draw_ui(self.player_sat, self.selected_body, self.sas_enabled, self.throttle, self.player_torque,
                                  self.mission_start_time, self.simulation_time, self.fast_forward_rate, self.capture_state, self.capture_progress)
        elif self.state in (GameState.CLEAR, GameState.GAMEOVER):
            is_player_alive = self.player_sat in self.engine.bodies
            is_doomed_debri_alive = self.doomed_debri in self.engine.bodies

            if self.is_cinematic_mode and (is_player_alive or is_doomed_debri_alive): # シネマティックフェーズ
                if is_player_alive:
                    self.selected_body = self.player_sat
                else:
                    self.selected_body = self.doomed_debri
                self.tracking_camera.set_target_body(self.selected_body)
                self.tracking_camera.set_pixels_per_du(PIXELS_PER_DU * 3) # 地表面がギリギリ見えるように拡大

                # 画面を暗くせず（bg_alpha=0），理由と「見届けろ」というメッセージだけを出す．
                self.renderer.draw_overlay("Watch the re-entry...", self.end_reason, (255, 200, 50), bg_alpha=0)
            else:
                # 最終リザルト
                if self.state == GameState.CLEAR:
                    self.renderer.draw_overlay("MISSION SUCCESS", f"{self.end_reason} Press 'SPACE' to Restart.", (100, 255, 100))
                else:
                    self.renderer.draw_overlay("MISSION FAILED", f"{self.end_reason} Press 'SPACE' to Restart.", (255, 50, 50))

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
