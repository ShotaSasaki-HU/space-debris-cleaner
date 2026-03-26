# physics/engine.py
import numpy as np
from typing import List

from physics.body import RigidBody

class GravityEngine:
    """
    物理シミュレーションのルールを司るエンジンクラス．
    速度ベルレ法を用いて，管理下にある全剛体の状態（位置・速度・加速度）を更新する．
    """
    
    def __init__(self, time_step: float):
        """
        Args:
            time_step (float): 1ステップで進める時間 dt (TU)
        """
        self.time_step: float = time_step
        self.bodies: List[RigidBody] = []

    def add_body(self, body: RigidBody) -> None:
        """シミュレーション空間に剛体を追加"""
        self.bodies.append(body)

    def _compute_accelerations(self) -> None:
        """
        現在の位置に基づき，全天体間の万有引力を計算し，各剛体の加速度を更新する内部メソッド．
        O(N^2)の多体問題（N-body problem）として一般化して実装．
        """
        # 1. 毎ステップ，過去の加速度をゼロにリセットする．
        for body in self.bodies:
            body.acceleration.fill(0.0)

        n = len(self.bodies)
        # 2. 全天体の組み合わせで重力を計算
        for i in range(n):
            if self.bodies[i].is_fixed:
                continue # 地球などの固定天体は加速度を計算しない．
                
            for j in range(n):
                if i == j:
                    continue # 自分自身との重力は計算しない．
                    
                # i から j へ向かう位置ベクトル
                r_vec = self.bodies[j].position - self.bodies[i].position
                distance_sq = np.dot(r_vec, r_vec) # 距離の2乗（np.dotは内積）
                
                if distance_sq == 0:
                    continue # 衝突時のゼロ除算によるクラッシュを回避

                distance = np.sqrt(distance_sq)
                
                # a = (G_CANONICAL * M_j / r^2) * (r_vec / r)
                acc_mag = (self.bodies[j].mass) / distance_sq # 万有引力による加速度の大きさ
                unit_r_vec = r_vec / distance # 単位位置ベクトル
                self.bodies[i].acceleration += acc_mag * unit_r_vec

    def initialize(self) -> None:
        """
        シミュレーション開始前に1度だけ呼び出す初期化メソッド．
        ベルレ法は最初のステップで「現在の加速度 a(t)」を必要とするため，初期位置に基づく加速度を事前計算しておく．
        """
        self._compute_accelerations()

    def step(self) -> None:
        """
        速度ベルレ法（Velocity Verlet）による1ステップの時間更新
        """
        dt = self.time_step
        
        # Step 1: 現在の加速度 a(t) を用いて，位置 x(t+dt) を更新．
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        for body in self.bodies:
            if not body.is_fixed:
                body.position += body.velocity * dt + 0.5 * body.acceleration * (dt ** 2)
                
        # Step 2: 現在の加速度 a(t) を退避させる．（Step 4の速度計算で使うため．）
        old_accelerations = [np.copy(body.acceleration) for body in self.bodies]
        
        # Step 3: 新しい位置 x(t+dt) における，新しい加速度 a(t+dt) を計算．
        self._compute_accelerations()
        
        # Step 4: 古い加速度 a(t) と新しい加速度 a(t+dt) の平均を用いて，速度 v(t+dt) を更新．
        # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        for i, body in enumerate(self.bodies):
            if not body.is_fixed:
                body.velocity += 0.5 * (old_accelerations[i] + body.acceleration) * dt
