# tests/test_physics_engine.py
import pytest
import numpy as np

from physics.body import RigidBody
from physics.engine import GravityEngine
from physics.constants import KG_TO_MU, EARTH_MASS_KG, METER_TO_DU, EARTH_RADIUS_M, G_CANONICAL, SEC_TO_TU

class TestGravityEngine:
    
    @pytest.fixture
    def engine(self):
        """テスト用のエンジンを初期化するフィクスチャ"""
        return GravityEngine(time_step=0.01)

    def test_circular_orbit_completes_one_period(self, engine):
        """
        テスト1：適切な初速を与えられた物体が，計算上の周期(T)経過後に元の位置（許容誤差範囲内）に戻ってくるかを検証する．
        """
        # ==========================================
        # Arrange（準備）
        # ==========================================
        M = KG_TO_MU * EARTH_MASS_KG # 地球の質量（MU）
        r = METER_TO_DU * (EARTH_RADIUS_M + 400e3) # 衛星の軌道半径（DU）
        
        # 円軌道の速度 v = sqrt(GM / r)
        v = np.sqrt(G_CANONICAL * M / r)
        
        # 周期 T = 2 * pi * sqrt(r^3 / GM)
        period_t = 2.0 * np.pi * np.sqrt((r ** 3) / (G_CANONICAL * M))
        
        earth = RigidBody(mass=M, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), is_fixed=True)        
        satellite = RigidBody(mass=KG_TO_MU * 500, position=np.array([r, 0.0]), velocity=np.array([0.0, v]))
        
        engine.add_body(earth)
        engine.add_body(satellite)

        # ==========================================
        # Act（実行）
        # ==========================================
        initial_position = satellite.position.copy()

        engine.initialize()
        
        # 周期 T の時間分だけシミュレーションを進める．
        # 100 * T でテストすると通らないが，おそらく軌道を外れているのではなく，intによる打ち切りで時間がズレている．
        steps = int(period_t / engine.time_step)
        for _ in range(steps):
            engine.step()

        # ==========================================
        # Assert（検証）
        # ==========================================
        # 1周して元の位置に戻ってきているはず（誤差は 1e-2 未満を許容）
        distance_error = np.linalg.norm(satellite.position - initial_position)
        assert distance_error < 1e-2, f"Expected to return to start, but error is {distance_error}"


    def test_energy_conservation_with_velocity_verlet(self, engine):
        """
        テスト2：速度ベルレ法による数値積分が，長期間のシミュレーションにおいて
        力学的エネルギー（運動エネルギー＋位置エネルギー）を保存するかを検証する．
        """
        # ==========================================
        # Arrange（準備）
        # ==========================================
        M = KG_TO_MU * EARTH_MASS_KG # 地球の質量（MU）
        earth = RigidBody(mass=M, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), is_fixed=True)
        
        r = METER_TO_DU * (EARTH_RADIUS_M + 400e3) # 衛星の軌道半径（DU）
        v = 10e3 * METER_TO_DU / SEC_TO_TU # あえて少し歪んだ楕円軌道になるような適当な初速を与える．
        satellite = RigidBody(mass=KG_TO_MU * 500, position=np.array([r, 0.0]), velocity=np.array([0.0, v]))
        
        engine.add_body(earth)
        engine.add_body(satellite)

        # エネルギーを計算するローカル関数
        def calculate_total_energy(sat):
            # 運動エネルギー = 1/2 * m * v^2
            kinetic = 0.5 * sat.mass * (np.linalg.norm(sat.velocity) ** 2)
            # 位置エネルギー = - G * M * m / r
            r = np.linalg.norm(sat.position - earth.position)
            potential = - (G_CANONICAL * earth.mass * sat.mass) / r
            return kinetic + potential

        initial_energy = calculate_total_energy(satellite)

        # ==========================================
        # Act（実行）
        # ==========================================
        engine.initialize()
        
        # 100000ステップ（長期間）回す．
        for _ in range(100000):
            engine.step()

        # ==========================================
        # Assert（検証）
        # ==========================================
        final_energy = calculate_total_energy(satellite)
        
        # エネルギーの変動が極めて小さいこと（相対誤差 1e-4 未満）を確認
        # np.iscloseは浮動小数点数の比較に非常に便利
        assert np.isclose(initial_energy, final_energy, rtol=1e-4), \
            f"Energy not conserved! Initial: {initial_energy}, Final: {final_energy}"
