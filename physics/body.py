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
        is_fixed: bool = False
    ):
        """
        Args:
            mass (float): 質量 (MU)
            position (np.ndarray): 2次元の位置ベクトル [x, y] (DU)
            velocity (np.ndarray): 2次元の速度ベクトル [vx, vy] (DU/TU)
            is_fixed (bool): Trueの場合，物理エンジンによる位置更新を受け付けない．（地球用）
        """
        self.mass: float = mass
        # float64型を明示し，計算精度を確保．
        self.position: np.ndarray = np.array(position, dtype=np.float64)
        self.velocity: np.ndarray = np.array(velocity, dtype=np.float64)
        
        # 加速度は物理エンジンが計算するため，初期値はゼロベクトル．
        self.acceleration: np.ndarray = np.zeros(2, dtype=np.float64)
        
        self.is_fixed: bool = is_fixed

    def apply_force(self, force_vector: np.ndarray) -> None:
        """
        物体に力を加え，加速度を更新する（スラスター噴射などで使用予定）．
        F = ma より，a = F / m．
        """
        if not self.is_fixed and self.mass > 0:
            self.acceleration += force_vector / self.mass
