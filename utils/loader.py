import json
import numpy as np
from physics.body import RigidBody
from physics.constants import KG_TO_MU, METER_TO_DU, SEC_TO_TU

class LevelLoader:
    """外部ファイルからステージ情報（デブリなど）を読み込むローダークラス"""
    
    @staticmethod
    def load_debris_from_json(filepath: str) -> list[RigidBody]:
        """
        SI単位系で記述されたJSONを読み込み，カノニカル単位系に変換してRigidBodyのリストを生成する．
        """
        bodies = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                # 質量の変換 (kg -> MU)
                mass_cano = item["mass_kg"] * KG_TO_MU
                
                # 位置の変換 (m -> DU)
                pos_si = np.array(item["position_m"], dtype=np.float64)
                pos_cano = pos_si * METER_TO_DU
                
                # 速度の変換 (m/s -> DU/TU)
                vel_si = np.array(item["velocity_m_s"], dtype=np.float64)
                vel_cano = vel_si * (METER_TO_DU / SEC_TO_TU)
                
                # 慣性モーメントの変換 (kg*m^2 -> MU*DU^2)
                inertia_cano = item["moment_of_inertia_kg_m2"] * KG_TO_MU * (METER_TO_DU ** 2)
                
                # 角度の変換 (deg -> rad)
                angle = np.deg2rad(item["angle_deg"])

                # 角速度の変換 (deg/s -> rad/TU)
                ang_vel_deg_s = item["angular_velocity_deg_s"]
                ang_vel_rad_s = np.deg2rad(ang_vel_deg_s)
                ang_vel_cano = ang_vel_rad_s / SEC_TO_TU
                
                # 寸法の変換 (m -> DU)
                width_cano = item["width_m"] * METER_TO_DU
                height_cano = item["height_m"] * METER_TO_DU

                # インスタンス生成
                debri = RigidBody(
                    mass=mass_cano,
                    position=pos_cano,
                    velocity=vel_cano,
                    moment_of_inertia=inertia_cano,
                    angle=angle,
                    angular_velocity=ang_vel_cano,
                    image_path=item["image_path"],
                    real_width_du=width_cano,
                    real_height_du=height_cano,
                    draw_fixed_size_px=item["draw_fixed_size_px"]
                )
                bodies.append(debri)
                
        except FileNotFoundError:
            print(f"Error: 構成ファイル {filepath} が見つかりません．")
        except KeyError as e:
            print(f"Error: JSONデータのキー {e} が不足しています．")
            
        return bodies
