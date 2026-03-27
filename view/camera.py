# view/camera.py
import numpy as np
from typing import Tuple

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
