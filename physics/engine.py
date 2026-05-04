# physics/engine.py
import numpy as np
from typing import List, Dict
import copy
from dataclasses import dataclass

from physics.body import RigidBody

@dataclass
class CollisionEvent:
    """
    衝突の結果をメインループ（コントローラー層）に伝達するためのデータ（DTO）クラス．
    """
    body1: RigidBody
    body2: RigidBody
    impact_speed_cano: float # 衝突時の相対速度（カノニカル単位系）
    body1_destroyed: bool    # 破壊されたかどうかのフラグ
    body2_destroyed: bool

class GravityEngine:
    """
    物理シミュレーションのルールを司るエンジンクラス．
    速度ベルレ法を用いて，管理下にある全剛体の状態（位置・速度・加速度）を更新する．
    """
    def __init__(self, time_step: float, surface_radius_du: float, atmosphere_radius_du: float):
        """
        Args:
            time_step (float): 1ステップで進める時間 dt (TU)
        """
        self.time_step: float = time_step
        self.bodies: List[RigidBody] = []

        self.restitution_coefficient = 0.6 # 反発係数

        self.surface_radius_du: float = surface_radius_du
        self.atmosphere_radius_du: float = atmosphere_radius_du

    def add_body(self, body: RigidBody) -> None:
        """シミュレーション空間に剛体を追加"""
        self.bodies.append(body)
    
    def remove_body(self, body: RigidBody) -> None:
        """シミュレーション空間から剛体を削除"""
        self.bodies.remove(body)
    
    def _resolve_collisions(self, target_bodies: List[RigidBody]) -> List[CollisionEvent]:
        """
        剛体同士の重なりを検知し，位置補正と速度ベクトルの弾性衝突演算を行う．
        """
        events = []
        n = len(target_bodies)

        # 総当たり判定
        for i in range(n):
            for j in range(i + 1, n):
                b1 = target_bodies[i]
                b2 = target_bodies[j]

                r1 = b1.get_collision_radius()
                r2 = b2.get_collision_radius()
                if r1 == 0 and r2 == 0: continue

                r_vec = b2.position - b1.position
                dist = np.linalg.norm(r_vec)
                min_dist = r1 + r2

                if dist < min_dist:
                    n_vec = r_vec / dist if dist != 0 else np.array([1.0, 0.0]) # 単位法線ベクトル

                    # --- 重なり解消の位置補正ココカラ ---

                    overlap = min_dist - dist
                    total_mass = b1.mass + b2.mass

                    if not b1.is_fixed and not b2.is_fixed:
                        b1.position -= n_vec * (overlap * (b2.mass / total_mass))
                        b2.position += n_vec * (overlap * (b1.mass / total_mass))
                    elif b1.is_fixed and not b2.is_fixed:
                        b2.position += n_vec * overlap
                    elif not b1.is_fixed and b2.is_fixed:
                        b1.position -= n_vec * overlap
                    else:
                        continue

                    # --- 重なり解消の位置補正ココマデ ---

                    # --- 速度ベクトルの更新ココカラ ---

                    v_rel = b2.velocity - b1.velocity
                    v_rel_n = np.dot(v_rel, n_vec) # 法線方向の相対速度（正射影の大きさ）

                    # 既に離れようとしている場合は速度を変えない．
                    if v_rel_n > 0: continue

                    inv_m1 = 1.0 / b1.mass if not b1.is_fixed else 0.0
                    inv_m2 = 1.0 / b2.mass if not b2.is_fixed else 0.0
                    j = -(1.0 + self.restitution_coefficient) * v_rel_n / (inv_m1 + inv_m2)

                    # 速度ベクトルの更新（v1'とv2'を反発係数の式に代入してみると確認できるよ．）
                    if not b1.is_fixed:
                        b1.velocity -= (j * inv_m1) * n_vec
                    if not b2.is_fixed:
                        b2.velocity += (j * inv_m2) * n_vec

                    # --- 速度ベクトルの更新ココマデ ---

                    # --- 破壊判定ココカラ ---
                    # 参考：https://physics-school.com/two-body-energy/

                    reduced_mass = (b1.mass * b2.mass) / (b1.mass + b2.mass) # 換算質量
                    impact_speed = abs(v_rel_n) # 衝突前の相対速度の大きさ

                    # 相対運動エネルギーの減少量（完全弾性衝突ならe=1より，dE=0となる．）
                    delta_energy = 0.5 * reduced_mass * (1 - (self.restitution_coefficient ** 2)) * (impact_speed ** 2)

                    # 個別に破壊判定
                    b1_destroyed = delta_energy > b1.crash_tolerance_cano
                    b2_destroyed = delta_energy > b2.crash_tolerance_cano
                    # print(f"dE: {delta_energy}")
                    # print(f"b1.crash_tolerance_joules: {b1.crash_tolerance_cano}")
                    # print(f"b2.crash_tolerance_joules: {b2.crash_tolerance_cano}")

                    # --- 破壊判定ココマデ ---

                    events.append(CollisionEvent(b1, b2, impact_speed, b1_destroyed, b2_destroyed))

        return events

    def _compute_accelerations_for(self, target_bodies: List[RigidBody]) -> None:
        """
        指定された剛体リストに対して，万有引力と外部推力による加速度を計算する．
        """
        for body in target_bodies:
            # 万有引力の計算前に，プレイヤーからの推力を加速度にセットする．
            if body.mass > 0:
                body.acceleration = body.applied_force / body.mass
                body.angular_acceleration = body.applied_torque / body.moment_of_inertia
            else:
                body.acceleration.fill(0.0)
                body.angular_acceleration = 0.0

        # 万有引力
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
        
        # 大気抵抗
        self._apply_aerodynamic_drag(target_bodies, self.surface_radius_du, self.atmosphere_radius_du)
    
    def _apply_aerodynamic_drag(self, target_bodies: List[RigidBody], surface_radius_du: float, atmosphere_radius_du: float):
        """
        大気圏内の物体に大気抵抗を適用する．
        """
        for body in target_bodies:
            if body.is_fixed: continue
            
            r_mag = np.linalg.norm(body.position)
            if r_mag < atmosphere_radius_du:
                v_mag = np.linalg.norm(body.velocity)
                if v_mag > 0:
                    # 高度に基づく簡易大気密度ファクター（地表：1.0 〜 大気圏の境界：0.0）
                    body_alt_du = r_mag - surface_radius_du
                    atmosphere_alt_du = atmosphere_radius_du - surface_radius_du
                    density_factor = max(0.0, 1.0 - (body_alt_du / atmosphere_alt_du))
                    
                    # 速度の2乗に比例して強くなる抵抗力による加速度
                    # ※ 係数はゲーム的なチューニング値である．必要に応じて増減させる．
                    # カノニカル単位系により極小になっている質量で力を割ると，値が爆発してフリーズしやすい．よって，係数にまとめて委ねている．
                    drag_acc_mag = density_factor * (v_mag ** 2) * 0.5
                    drag_dir = -body.velocity / v_mag
                    
                    body.acceleration += drag_dir * drag_acc_mag

    def _step_bodies(self, target_bodies: List[RigidBody], dt: float, includes_collision: bool) -> List[CollisionEvent]:
        """
        指定された剛体リストと時間刻み幅で，並進と回転の速度ベルレ法を実行し，衝突も解決する．
        """
        # Step 1: 位置更新
        for body in target_bodies:
            if not body.is_fixed:
                body.position += body.velocity * dt + 0.5 * body.acceleration * (dt ** 2)
                body.angle += body.angular_velocity * dt + 0.5 * body.angular_acceleration * (dt ** 2)
                
        # Step 2: 加速度退避
        old_accs = [b.acceleration.copy() for b in target_bodies]
        old_angle_accs = [b.angular_acceleration for b in target_bodies] # floatはイミュータブル
        
        # Step 3: 新しい加速度計算
        self._compute_accelerations_for(target_bodies)
        
        # Step 4: 速度・角速度更新
        for i, body in enumerate(target_bodies):
            if not body.is_fixed:
                body.velocity += 0.5 * (old_accs[i] + body.acceleration) * dt
                body.angular_velocity += 0.5 * (old_angle_accs[i] + body.angular_acceleration) * dt
            
            body.clear_applied_forces() # スラスター入力のリセット
        
        # Step 5: 衝突の解決（速度更新後に行うことで運動量が保存される？）
        collision_events = self._resolve_collisions(target_bodies) if includes_collision else []

        # Step 6: docked_bodyへ速度ベクトルを共有（空力加熱エフェクトのため）
        for body in target_bodies:
            if body.docked_body:
                body.docked_body.velocity = body.velocity
                break

        return collision_events

    def initialize(self) -> None:
        """
        シミュレーション開始前に1度だけ呼び出す初期化メソッド．
        ベルレ法は最初のステップで「現在の加速度 a(t)」を必要とするため，初期位置に基づく加速度を事前計算しておく．
        """
        self._compute_accelerations_for(self.bodies)

    def step(self) -> List[CollisionEvent]:
        """1ステップ計算を進め，発生した衝突イベントのリストを返す．"""
        return self._step_bodies(self.bodies, self.time_step, includes_collision=True)
    
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

        # 予測開始時はすべて予測をアクティブにセット
        for tb in temp_bodies:
            tb.is_active_for_prediction = True
        
        # 結果を格納する辞書（天体ごとに位置リストを持つ．）
        # id()を使って個々のRigidBodyを識別
        predictions: Dict[int, List[np.ndarray]] = {id(b): [b.position.copy()] for b in self.bodies}

        # 仮想宇宙の時間ステップ
        steps = int(future_duration / dt_prediction)
        
        self._compute_accelerations_for(temp_bodies) # 初期の加速度

        # 仮想宇宙でのシミュレーションループ
        for _ in range(steps):
            # まだ生きている（地表に到達していない）剛体だけを抽出して計算を進める
            active_temp_bodies = [b for b in temp_bodies if b.is_active_for_prediction]
            
            # 生き残りがゼロなら予測ループを早期終了
            if not active_temp_bodies:
                break

            self._step_bodies(active_temp_bodies, dt_prediction, includes_collision=False)

            for orig_body, temp_body in zip(self.bodies, temp_bodies): # 2つのリストの順番が完全に一致している前提
                if orig_body.is_fixed:
                    continue
                    
                # すでに地表に到達して計算から除外されている場合はスキップ
                if not getattr(temp_body, 'is_active_for_prediction', True):
                    continue

                # 地表（＋わずかな猶予）を下回ったかチェック
                if np.linalg.norm(temp_body.position) <= self.surface_radius_du + 0.001:
                    temp_body.is_active_for_prediction = False # 次のステップから除外
                    # 最後に地表スレスレの座標を記録して終了
                    norm_pos = temp_body.position / np.linalg.norm(temp_body.position)
                    predictions[id(orig_body)].append(norm_pos * self.surface_radius_du)
                else:
                    # 現実の剛体のIDをキーにして，未来の剛体の位置を記録する．
                    predictions[id(orig_body)].append(temp_body.position.copy())
  
        return predictions
    
    def set_time_step(self, time_step: float) -> None: self.time_step = time_step
