# physics/body.py
import numpy as np

from physics.constants import METER_TO_DU, SEC_TO_TU

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
        angular_velocity: float = 0.0,
        is_fixed: bool = False,
        image_path: str = None,
        real_width_du: float = 0.0,
        real_height_du: float = 0.0,
        draw_fixed_size_px: int = None,
        isp_sec: float = 220.0
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
            isp_sec (float): エンジンの比推力（秒）
        """
        self.mass: float = mass
        self.position: np.ndarray = np.array(position, dtype=np.float64) # float64型を明示し，計算精度を確保．
        self.velocity: np.ndarray = np.array(velocity, dtype=np.float64)
        self.acceleration: np.ndarray = np.zeros(2, dtype=np.float64) # 加速度は物理エンジンが計算するため，初期値はゼロベクトル．

        self.moment_of_inertia = moment_of_inertia
        self.angle = angle
        self.angular_velocity = angular_velocity
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

        # 衝突半径
        if is_fixed:
            self.collision_radius: float = min(real_width_du, real_height_du) / 2.0
        else:
            self.collision_radius: float = (min(real_width_du, real_height_du) / 2.0) * 0.8
        self.crash_tolerance_cano: float = mass / 1e7 # 構造強度（自身の質量の1/N倍のエネルギーまで耐えられると仮定．SI単位系ならジュール．）

        # 結合物理用の拡張パラメータ
        self.visual_offset_local = np.array([0.0, 0.0]) # 新たな重心に対する本来の画像の描画オフセット
        self.docked_body: RigidBody = None              # 捕獲したデブリのインスタンス
        self.docked_offset_local = np.array([0.0, 0.0]) # 新たな重心に対するデブリの描画オフセット
        self.docked_rel_angle = 0.0                     # 自機に対するデブリの相対角度
        self._original_mass = mass                      # オリジナル諸元の保存用
        self._original_inertia = moment_of_inertia      # オリジナル諸元の保存用

        # 燃料
        self.max_propellant_mass = mass * 0.5
        self.propellant_mass = self.max_propellant_mass

        exhaust_velocity_si = isp_sec * 9.80665 # Ispから実効排気速度（m/s）を計算（g0 = 9.80665 m/s^2）
        exhaust_velocity_cano = exhaust_velocity_si * (METER_TO_DU / SEC_TO_TU) # 実効排気速度をカノニカル単位系（DU/TU）に変換
        # 消費係数は排気速度の逆数（1 / v_e）
        if exhaust_velocity_cano > 0:
            self.fuel_consumption_rate = 1.0 / exhaust_velocity_cano
        else:
            self.fuel_consumption_rate = 0.0

    def apply_local_force(self, force_local_x: float, force_local_y: float, total_force_mag: float, dt: float) -> None:
        """
        機体のローカル座標系で推力を加える．（W/S, A/Dキー用）
        Args:
            force_local_x (float): Sが負，Wが正．
            force_local_y (float): Aが正，Dが負．PyGameの座標系に合わせる場合は反転に注意．
        """
        if self.is_fixed or self.mass <= 0:
            return
        
        if total_force_mag > 0:
            # 燃料を消費．足りなければ推力は発生しない．
            if not self.consume_fuel(total_force_mag, dt):
                return

        # 回転行列を用いてローカル推力をワールド推力に変換
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        
        world_force_x = force_local_x * cos_theta - force_local_y * sin_theta
        world_force_y = force_local_x * sin_theta + force_local_y * cos_theta
        
        self.applied_force += np.array([world_force_x, world_force_y])
    
    def apply_local_force_at_offset(self, force_local_x: float, force_local_y: float,
                                    offset_x_local: float, offset_y_local: float, total_force_mag: float, dt: float):
        """重心からズレた位置にローカル座標系の力を加える．（トルクも発生）"""
        if total_force_mag > 0:
            # 燃料を消費．足りなければ推力は発生しない．
            if not self.consume_fuel(total_force_mag, dt):
                return

        cos_t = np.cos(self.angle)
        sin_t = np.sin(self.angle)

        # 力のワールド変換
        fx_world = force_local_x * cos_t - force_local_y * sin_t
        fy_world = force_local_x * sin_t + force_local_y * cos_t
        self.applied_force += np.array([fx_world, fy_world])

        # 力の作用点のワールド変換（重心からの位置ベクトルr）
        r_world_x = offset_x_local * cos_t - offset_y_local * sin_t
        r_world_y = offset_x_local * sin_t + offset_y_local * cos_t
        
        # トルクの発生（外積: r × F）
        self.applied_torque += (r_world_x * fy_world - r_world_y * fx_world)
    
    def consume_fuel(self, force_mag: float, dt: float) -> bool:
        """
        指定された推力と時間に基づいて燃料を消費する．
        Returns:
            (bool): 燃料が足りていればTrue，空ならFalse．
        """
        if self.propellant_mass <= 0:
            return False
            
        # 消費量 = 推力の大きさ * 消費係数 * 時間
        dm = force_mag * self.fuel_consumption_rate * dt
        
        self.propellant_mass -= dm
        self.mass -= dm
        
        if self.propellant_mass <= 0:
            self.propellant_mass = 0.0
            return False
            
        return True

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
    
    def get_velo_from_imu(self) -> np.ndarray:
        """
        IMUセンサーが搭載されている位置（元の重心）における，機体ローカル座標系（前後・左右）の速度ベクトルを計算する．
        """
        # 1. 重心の並進速度（ワールド）をローカル座標に変換
        cos_t = np.cos(self.angle)
        sin_t = np.sin(self.angle)
        v_com_local_x = self.velocity[0] * cos_t + self.velocity[1] * sin_t
        v_com_local_y = -self.velocity[0] * sin_t + self.velocity[1] * cos_t
        
        # 2. IMUの位置（ローカルでの重心からのオフセット）
        r_x = self.visual_offset_local[0]
        r_y = self.visual_offset_local[1]
        
        # 3. 自転による接線速度（v_tan = ω × r）をローカル座標系で直接計算
        # （角速度 ω が正のとき，+X軸上の点は+Y方向へ，+Y軸上の点は-X方向へ動く．）
        omega = self.angular_velocity
        v_tan_local_x = -omega * r_y
        v_tan_local_y = omega * r_x
        
        # 4. 重心速度と接線速度を合成して，IMU位置の純粋なローカル速度とする．
        v_imu_local_x = v_com_local_x + v_tan_local_x
        v_imu_local_y = v_com_local_y + v_tan_local_y
        
        return np.array([v_imu_local_x, v_imu_local_y])
    
    def get_acc_from_imu(self) -> np.array:
        if self.mass > 0:
            # 1. 重心の並進加速度 (ワールド)
            acc_com_world = self.last_applied_force / self.mass
            
            # 2. ローカル座標への変換
            cos_t = np.cos(self.angle)
            sin_t = np.sin(self.angle)
            acc_com_x = acc_com_world[0] * cos_t + acc_com_world[1] * sin_t
            acc_com_y = -acc_com_world[0] * sin_t + acc_com_world[1] * cos_t
            
            # 3. IMUの位置（ローカルでの重心からのオフセット）
            r_x = self.visual_offset_local[0]
            r_y = self.visual_offset_local[1]
            
            # 4. 接線加速度 (a_tan = α × r)
            alpha_si = self.angular_acceleration
            a_tan_x = -alpha_si * r_y
            a_tan_y = alpha_si * r_x
            
            # 5. 向心加速度 (a_cen = -ω^2 * r)
            omega_si = self.angular_velocity
            a_cen_x = -(omega_si**2) * r_x
            a_cen_y = -(omega_si**2) * r_y
            
            # 全てを合成して初めて「IMUセンサーの生値」になる．
            acc_x = acc_com_x + a_tan_x + a_cen_x
            acc_y = acc_com_y + a_tan_y + a_cen_y

            return np.array([acc_x, acc_y])
        else:
            return np.array([0.0, 0.0])
    
    def get_position(self) -> np.ndarray: return self.position
    def get_collision_radius(self) -> float: return self.collision_radius
    def get_angular_velocity(self) -> float: return self.angular_velocity
    def get_angular_acceleration(self) -> float: return self.angular_acceleration
