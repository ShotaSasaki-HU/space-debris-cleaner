# view/camera.py
import numpy as np
from typing import Tuple
from physics.body import RigidBody

class Camera:
    """
    物理空間（カノニカル単位系）と画面空間（ピクセル）の座標変換を行うクラス．
    MVCアーキテクチャにおけるViewの基盤．
    """
    def __init__(self, screen_width: int, screen_height: int, pixels_per_du: float):
        """
        Args:
            screen_width (int): 画面の幅（ピクセル）
            screen_height (int): 画面の高さ（ピクセル）
            pixels_per_du (float): 1 DU（地球半径）を画面上で何ピクセルとして描画するか．（ズーム倍率）
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.pixels_per_du = pixels_per_du
        
        # 画面の中心座標（ここを物理空間の原点とする．）
        self.center_x = screen_width // 2
        self.center_y = screen_height // 2

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """
        物理エンジンの座標（カノニカル単位系）を，PyGameの画面座標（ピクセル）に変換する．
        Y軸の反転処理をここで行う．
        """
        screen_x = self.center_x + int(world_pos[0] * self.pixels_per_du) # X座標 = 画面中央 + (物理X * スケール)
        screen_y = self.center_y - int(world_pos[1] * self.pixels_per_du) # Y座標 = 画面中央 - (物理Y * スケール)
        
        return (screen_x, screen_y)
    
    def set_pixels_per_du(self, pixels_per_du: float) -> None: self.pixels_per_du = pixels_per_du

    def get_pixels_per_du(self) -> float: return self.pixels_per_du

class RelativeCamera:
    """
    近傍運用用のカメラ．ターゲットを画面中央に固定し，地球を下に回転させて描画する．
    """
    def __init__(self, screen_width: int, screen_height: int, pixels_per_du: float):
        self.center_x = screen_width // 2
        self.center_y = screen_height // 2
        # マクロ視点の数千〜数万倍のズーム倍率を設定する
        self.pixels_per_du = pixels_per_du 
        self.target_body: RigidBody = None

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        if self.target_body is None:
            return (self.center_x, self.center_y)

        # 1. ターゲットからの相対位置ベクトル
        delta_pos = world_pos - self.target_body.position

        # 2. ターゲットの「地球に対する角度（位相）」を計算
        # 地球は原点なので，ターゲットの絶対座標からそのまま角度が出る．
        theta = np.atan2(self.target_body.position[1], self.target_body.position[0])

        # 3. 相対位置ベクトルを，ターゲットの角度分だけ「逆回転」させる．
        # x_local: 動径方向（外側がプラス，地球側がマイナス）
        # y_local: 進行方向
        x_local = delta_pos[0] * np.cos(theta) + delta_pos[1] * np.sin(theta)
        y_local = -delta_pos[0] * np.sin(theta) + delta_pos[1] * np.cos(theta)

        # 4. PyGameの画面座標にマッピング
        # 地球の重力方向は常に「画面の下」にする．
        screen_x = self.center_x - int(y_local * self.pixels_per_du)
        screen_y = self.center_y - int(x_local * self.pixels_per_du)

        return (screen_x, screen_y)
    
    def set_pixels_per_du(self, pixels_per_du: float) -> None: self.pixels_per_du = pixels_per_du
    def set_target_body(self, target_body: RigidBody) -> None: self.target_body = target_body
    
    def get_pixels_per_du(self) -> float: return self.pixels_per_du
    def get_target_body(self) -> RigidBody: return self.target_body
