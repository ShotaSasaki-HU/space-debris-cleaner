# physics/engine.py
import numpy as np
from typing import List, Dict
import copy

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

    def _compute_accelerations_for(self, target_bodies: List[RigidBody]) -> None:
        """
        全天体間の万有引力を計算し，各剛体の加速度を更新する内部メソッド．
        """
        for body in target_bodies:
            body.acceleration.fill(0.0)

        n = len(target_bodies)
        for i in range(n):
            if target_bodies[i].is_fixed: continue

            for j in range(n):
                if i == j: continue

                r_vec = target_bodies[j].position - target_bodies[i].position
                distance_sq = np.dot(r_vec, r_vec)
                if distance_sq == 0: continue

                acc_mag = target_bodies[j].mass / distance_sq # 万有引力による加速度の大きさ
                unit_r_vec = r_vec / np.sqrt(distance_sq) # 単位位置ベクトル
                target_bodies[i].acceleration += acc_mag * unit_r_vec

    def _step_bodies(self, target_bodies: List[RigidBody], dt: float) -> None:
        """
        指定された剛体リストと時間刻み幅で，速度ベルレ法の1ステップを実行する．
        """
        # Step 1: 位置更新
        for body in target_bodies:
            if not body.is_fixed:
                body.position += body.velocity * dt + 0.5 * body.acceleration * (dt ** 2)
                
        # Step 2: 加速度退避
        old_accels = [b.acceleration.copy() for b in target_bodies]
        
        # Step 3: 新しい加速度計算
        self._compute_accelerations_for(target_bodies)
        
        # Step 4: 速度更新
        for i, body in enumerate(target_bodies):
            if not body.is_fixed:
                body.velocity += 0.5 * (old_accels[i] + body.acceleration) * dt

    def initialize(self) -> None:
        """
        シミュレーション開始前に1度だけ呼び出す初期化メソッド．
        ベルレ法は最初のステップで「現在の加速度 a(t)」を必要とするため，初期位置に基づく加速度を事前計算しておく．
        """
        self._compute_accelerations_for(self.bodies)

    def step(self) -> None:
        self._step_bodies(self.bodies, self.time_step)
    
    def predict_trajectories(self, future_duration: float, dt_prediction: float) -> Dict[int, List[np.ndarray]]:
        """
        現在の全天体の状態に基づき，未来の軌道を予測する．
        
        Args:
            future_duration (float): どれくらい未来まで予測するか．（TU）
            dt_prediction (float): 予測計算の時間刻み．ゲーム本体より粗くて良い．（TU）
        Returns:
            Dict[int, List[np.ndarray]]: RigidBodyのidをキー，未来の位置リスト（DU）を値とする辞書．
        """
        # 現在の状態を「仮想宇宙」にディープコピーする．これにより，ゲーム本体の状態を汚さずに計算できる．
        temp_bodies = copy.deepcopy(self.bodies)
        
        # 結果を格納する辞書（天体ごとに位置リストを持つ．）
        # id()を使って個々のRigidBodyを識別
        predictions: Dict[int, List[np.ndarray]] = {id(b): [b.position.copy()] for b in self.bodies}

        # 仮想宇宙の時間ステップ
        steps = int(future_duration / dt_prediction)
        
        self._compute_accelerations_for(temp_bodies) # 初期の加速度

        # 仮想宇宙でのシミュレーションループ
        for _ in range(steps):
            self._step_bodies(temp_bodies, dt_prediction)

            for orig_body, temp_body in zip(self.bodies, temp_bodies): # 2つのリストの順番が完全に一致している前提
                if not orig_body.is_fixed:
                    # 現実の剛体のIDをキーにして，未来の剛体の位置を記録する．
                    predictions[id(orig_body)].append(temp_body.position.copy())
  
        return predictions
