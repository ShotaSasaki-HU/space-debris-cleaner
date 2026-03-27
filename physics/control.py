# physics/control.py
import math

class PIDController:
    """
    宇宙機用のPIDコントローラー
    """
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute_torque(self, current_angle: float, target_angle: float, dt_tu: float) -> float:
        """
        現在の状態から，目標角度に向けるためのトルク（回転力）を計算する．
        """
        # 角度の誤差を計算        
        # 角度の境界問題（-pi から pi）を解決し，常に「最短経路」の誤差にする．
        # 例：errorが +358度 なら -2度 に変換される．
        error = target_angle - current_angle
        error = (error + math.pi) % (2 * math.pi) - math.pi

        self.integral += error * dt_tu

        p_term = self.kp * error # P項：誤差に比例した加速
        i_term = self.ki * self.integral # I項：誤差の積分に比例
        d_term = self.kd * ((error - self.prev_error) / dt_tu) # D項：ブレーキ

        self.prev_error = error
        
        return p_term + i_term + d_term
