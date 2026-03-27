# physics/control.py
import math

class PDController:
    """
    宇宙機用のPD（比例・微分）コントローラー．
    """
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

    def compute_torque(self, current_angle: float, target_angle: float, current_angular_velocity: float) -> float:
        """
        現在の状態から，目標角度に向けるためのトルク（回転力）を計算する．
        """
        # 角度の誤差を計算
        error = target_angle - current_angle
        
        # 角度の境界問題（-pi から pi）を解決し，常に「最短経路」の誤差にする．
        # 例：errorが +358度 なら -2度 に変換される．
        error = (error + math.pi) % (2 * math.pi) - math.pi

        # PD制御の計算（P項：誤差に比例した加速，D項：現在の角速度に対するブレーキ）
        # 宇宙空間ではD項が逆噴射の役割を果たすため極めて重要
        p_term = self.kp * error
        d_term = -self.kd * current_angular_velocity
        # print(p_term + d_term)
        
        return p_term + d_term
