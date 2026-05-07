# view/camera.py
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from physics.body import RigidBody
import pygame

class Camera(ABC):
    """
    物理空間（カノニカル単位系）と画面空間（ピクセル）の座標変換を行う抽象基底クラス．
    Strategyパターンのインターフェースとして機能する．
    """
    def __init__(self, screen: pygame.Surface, pixels_per_du: float):
        """
        Args:
            screen (pygame.Surface): スクリーン
            pixels_per_du (float): 1 DU（地球半径）を画面上で何ピクセルとして描画するか．
        """
        self.update_screen_size(screen=screen)
        self._pixels_per_du = pixels_per_du
    
    @abstractmethod
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """このメソッドは各具象Strategy（サブクラス）で実装しなければエラーとなる．"""
        pass
    
    def update_screen_size(self, screen: pygame.Surface) -> None:
        """ウィンドウサイズ変更時の再計算"""
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
    
    @property
    def pixels_per_du(self) -> float:
        return self._pixels_per_du

    @pixels_per_du.setter
    def pixels_per_du(self, value: float) -> None:
        self._pixels_per_du = value

class EarthCamera(Camera):
    """地球中心のカメラ（具象Strategy1）"""

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """
        物理エンジンの座標（カノニカル単位系）を，PyGameの画面座標（ピクセル）に変換する．
        Y軸の反転処理をここで行う．
        """
        screen_x = self.center_x + int(world_pos[0] * self.pixels_per_du) # X座標 = 画面中央 + (物理X * スケール)
        screen_y = self.center_y - int(world_pos[1] * self.pixels_per_du) # Y座標 = 画面中央 - (物理Y * スケール)
        return (screen_x, screen_y)

class RelativeCamera(Camera):
    """
    近傍運用用の追従カメラ（具象Strategy2）
    ターゲットを画面中央に固定し，地球を下に回転させて描画する．
    """
    def __init__(self, screen: pygame.Surface, pixels_per_du: float):
        super().__init__(screen, pixels_per_du)
        self._target_body: Optional[RigidBody] = None # RelativeCameraだけの専用プロパティ（基底に実装するとISP違反）

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
    
    @property
    def target_body(self) -> Optional[RigidBody]:
        return self._target_body

    @target_body.setter
    def target_body(self, body: RigidBody) -> None:
        self._target_body = body
