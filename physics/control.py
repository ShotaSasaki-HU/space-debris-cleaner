import math

class PIDController:
    """
    宇宙機用のPIDコントローラー（SI単位系ベース）
    """
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 1.0, max_integral: float = 10.0):
        self.kp = kp  # 単位: N·m / rad
        self.ki = ki  # 単位: N·m / (rad * s)
        self.kd = kd  # 単位: N·m / (rad / s)
        self.integral = 0.0
        self.max_integral = max_integral # 積分値の上限

    def compute_torque(self, current_angle: float, target_angle: float, current_angular_velocity: float, dt_sec: float) -> float:
        """
        目標角度に向けるためのトルク（N·m）を計算する．
        """
        error = target_angle - current_angle
        error = (error + math.pi) % (2 * math.pi) - math.pi

        # 誤差の積分
        self.integral += error * dt_sec
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)

        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = -self.kd * current_angular_velocity # 誤差の微分ではなく現在の角速度を使用．微分値は，角度の境界でスパイクを起こすため．

        return p_term + i_term + d_term
