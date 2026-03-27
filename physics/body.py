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
        is_fixed: bool = False
    ):
        """
        Args:
            mass (float): 質量 (MU)
            position (np.ndarray): 2次元の位置ベクトル [x, y] (DU)
            velocity (np.ndarray): 2次元の速度ベクトル [vx, vy] (DU/TU)
            moment_of_inertia (float): 慣性モーメント
            angle (float): 角度（ラジアン）
            is_fixed (bool): Trueの場合，物理エンジンによる位置更新を受け付けない．（地球用）
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
        self.applied_torque = 0.0 # トルク（回転力）

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
        self.applied_force.fill(0.0)
        self.applied_torque = 0.0
