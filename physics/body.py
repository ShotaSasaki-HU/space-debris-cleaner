# physics/body.py
import numpy as np

class RigidBody:
    """
    宇宙空間に存在する剛体（地球，デブリ，衛星など）のデータモデル．
    全ての物理量はカノニカル単位系（DU, MU, TU）で保持されることを前提とする．
    """
    def __init__(
        self,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
        moment_of_inertia: float = 1.0,
        angle: float = 0.0,
        is_fixed: bool = False,
        image_path: str = None,
        real_width_du: float = 0.0,
        real_height_du: float = 0.0,
        draw_fixed_size_px: int = None
    ):
        """
        Args:
            mass (float): 質量 (MU)
            position (np.ndarray): 2次元の位置ベクトル [x, y] (DU)
            velocity (np.ndarray): 2次元の速度ベクトル [vx, vy] (DU/TU)
            moment_of_inertia (float): 慣性モーメント
            angle (float): 角度（ラジアン）
            is_fixed (bool): Trueの場合，物理エンジンによる位置更新を受け付けない．
            image_path (str): 画像ファイルのパス
            real_width_du (float): DUベースの実寸法（幅）
            real_height_du (float): DUベースの実寸法（高さ）
            draw_fixed_size_px (int): マクロ視点での一定サイズ(ピクセル)
        """
        self.mass: float = mass
        self.position: np.ndarray = np.array(position, dtype=np.float64) # float64型を明示し，計算精度を確保．
        self.velocity: np.ndarray = np.array(velocity, dtype=np.float64)
        self.acceleration: np.ndarray = np.zeros(2, dtype=np.float64) # 加速度は物理エンジンが計算するため，初期値はゼロベクトル．

        self.moment_of_inertia = moment_of_inertia
        self.angle = angle
        self.angular_velocity = 0.0
        self.angular_acceleration = 0.0
        
        self.is_fixed: bool = is_fixed

        # 力のバッファ（毎フレームControllerからセットされ，物理エンジンが消費する．）
        self.applied_force = np.zeros(2, dtype=np.float64) # ワールド座標系での力
        self.last_applied_force = np.zeros(2, dtype=np.float64) # Renderer用に直前のフレームで加えられた力を退避するバッファ
        self.applied_torque = 0.0 # トルク（回転力）

        # ビジュアル情報
        self.image_path = image_path
        self.real_width_du = real_width_du
        self.real_height_du = real_height_du
        self.draw_fixed_size_px = draw_fixed_size_px

        self.collision_radius: float = min(real_width_du, real_height_du) / 2.0 # 衝突半径
        self.crash_tolerance_cano: float = mass / 1e7 # 構造強度（自身の質量の1/N倍のエネルギーまで耐えられると仮定．SI単位系ならジュール．）

        # 結合物理用の拡張パラメータ
        self.visual_offset_local = np.array([0.0, 0.0]) # 新たな重心に対する本来の画像の描画オフセット
        self.docked_body: RigidBody = None              # 捕獲したデブリのインスタンス
        self.docked_offset_local = np.array([0.0, 0.0]) # 新たな重心に対するデブリの描画オフセット
        self.docked_rel_angle = 0.0                     # 自機に対するデブリの相対角度
        self._original_mass = mass                      # オリジナル諸元の保存用
        self._original_inertia = moment_of_inertia      # オリジナル諸元の保存用

    def apply_local_force(self, force_local_x: float, force_local_y: float) -> None:
        """
        機体のローカル座標系で推力を加える．（W/S, A/Dキー用）
        Args:
            force_local_x (float): Sが負，Wが正．
            force_local_y (float): Aが正，Dが負．PyGameの座標系に合わせる場合は反転に注意．
        """
        if self.is_fixed or self.mass <= 0:
            return

        # 回転行列を用いてローカル推力をワールド推力に変換
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        
        world_force_x = force_local_x * cos_theta - force_local_y * sin_theta
        world_force_y = force_local_x * sin_theta + force_local_y * cos_theta
        
        self.applied_force += np.array([world_force_x, world_force_y])
    
    def apply_local_force_at_offset(self, fx: float, fy: float, offset_x_local: float, offset_y_local: float):
        """重心からズレた位置にローカル座標系の力を加える．（トルクも発生）"""
        cos_t = np.cos(self.angle)
        sin_t = np.sin(self.angle)

        # 力のワールド変換
        fx_world = fx * cos_t - fy * sin_t
        fy_world = fx * sin_t + fy * cos_t
        self.applied_force += np.array([fx_world, fy_world])

        # 力の作用点のワールド変換（重心からの位置ベクトルr）
        r_world_x = offset_x_local * cos_t - offset_y_local * sin_t
        r_world_y = offset_x_local * sin_t + offset_y_local * cos_t
        
        # トルクの発生（外積: r × F）
        self.applied_torque += (r_world_x * fy_world - r_world_y * fx_world)

    def apply_torque(self, torque: float) -> None:
        """
        機体にトルク（回転力）を加える．QがCCW，EがCW．
        """
        if not self.is_fixed and self.moment_of_inertia > 0:
            self.applied_torque += torque

    def clear_applied_forces(self) -> None:
        """
        毎フレームの物理計算終了後に，スラスターの推力をゼロにリセットする．
        """
        self.last_applied_force = self.applied_force.copy()
        self.applied_force.fill(0.0)
        self.applied_torque = 0.0
    
    def dock_with(self, other: 'RigidBody') -> None: # 自己参照
        """対象の剛体を自身に結合し，重心・質量・慣性モーメント・速度を合成する．"""
        total_mass = self.mass + other.mass

        # 新しい重心（CoM）
        new_com = (self.mass * self.position + other.mass * other.position) / total_mass
        new_vel = (self.mass * self.velocity + other.mass * other.velocity) / total_mass

        r1_world = self.position - new_com
        r2_world = other.position - new_com

        # 「平行軸の定理」と「同じ軸まわりの慣性モーメントの足し合わせ」による新しい慣性モーメントの計算
        # I_new = (I_1 + (m_1 * r_1^2)) + (I_2 + (m_2 * r_2^2))
        new_inertia = (self.moment_of_inertia + self.mass * np.dot(r1_world, r1_world) +
                       other.moment_of_inertia + other.mass * np.dot(r2_world, r2_world))
        
        # 角運動量保存則による新しい角速度の計算
        def cross_2d(a, b): return a[0]*b[1] - a[1]*b[0]
        v1_rel = self.velocity - new_vel
        v2_rel = other.velocity - new_vel
        # 「自転の角運動量」＋「公転の角運動量（各物体の重心が基準点に対して移動している事による角運動量）」
        L_total = (self.moment_of_inertia * self.angular_velocity + other.moment_of_inertia * other.angular_velocity +
                   self.mass * cross_2d(r1_world, v1_rel) + other.mass * cross_2d(r2_world, v2_rel))
        new_omega = L_total / new_inertia

        # ローカル座標への変換関数
        cos_t = np.cos(-self.angle)
        sin_t = np.sin(-self.angle)
        def to_local(vec):
            return np.array([vec[0]*cos_t - vec[1]*sin_t, vec[0]*sin_t + vec[1]*cos_t])

        # 状態の更新と結合
        self._original_mass = self.mass
        self._original_inertia = self.moment_of_inertia
        self.docked_body = other
        self.visual_offset_local = to_local(r1_world)
        self.docked_offset_local = to_local(r2_world)
        self.docked_rel_angle = other.angle - self.angle

        self.mass = total_mass
        self.moment_of_inertia = new_inertia
        self.position = new_com
        self.velocity = new_vel
        self.angular_velocity = new_omega

        return
    
    def undock(self) -> 'RigidBody':
        """結合を解除し，元の物理パラメータに戻しつつターゲットを分離する．"""
        if not self.docked_body: return None

        cos_t = np.cos(self.angle)
        sin_t = np.sin(self.angle)
        def to_world(vec):
            return np.array([vec[0]*cos_t - vec[1]*sin_t, vec[0]*sin_t + vec[1]*cos_t])

        r1_world = to_world(self.visual_offset_local)
        r2_world = to_world(self.docked_offset_local)

        # 切り離し時の速度ベクトル（自転による接線速度 v = ω × r を加算）
        com_vel = self.velocity.copy()
        def calc_vel(r_vec):
            return com_vel + (self.angular_velocity * np.array([-r_vec[1], r_vec[0]]))

        com = self.position.copy()

        # 自機を元の状態に復元
        self.position = com + r1_world
        self.velocity = calc_vel(r1_world)
        self.mass = self._original_mass
        self.moment_of_inertia = self._original_inertia
        self.visual_offset_local = np.array([0.0, 0.0])

        # ターゲットの空間復帰
        other = self.docked_body
        other.position = com + r2_world
        other.velocity = calc_vel(r2_world)
        other.angle = self.angle + self.docked_rel_angle
        other.angular_velocity = self.angular_velocity

        self.docked_body = None
        return other
    
    def get_position(self) -> np.ndarray: return self.position
    def get_collision_radius(self) -> float: return self.collision_radius
