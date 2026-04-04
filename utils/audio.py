# utils/audio.py
import pygame

class ThrusterAudioManager:
    """
    スラスターの継続音と収束音を状態管理に基づいて再生するマネージャークラス
    """
    def __init__(self, loop_wav_path: str, shutoff_wav_path: str):
        if not pygame.mixer.get_init():
            pygame.mixer.init() # pygameのミキサー初期化
        
        pygame.mixer.set_num_channels(16) # チャンネル数

        self.sound_loop = pygame.mixer.Sound(loop_wav_path)
        self.sound_shutoff = pygame.mixer.Sound(shutoff_wav_path)

        self.sound_loop.set_volume(1.0)
        self.sound_shutoff.set_volume(1.0)

        # 各スラスターの状態（ON/OFF）およびループ音を鳴らしているチャンネルを保持．
        self.states: dict[str, bool] = {}
        self.channels: dict[str, pygame.mixer.Channel] = {}
    
    def update_thruster(self, thruster_id: str, is_firing: bool):
        """
        毎フレーム呼ばれ，噴射状態の変化を検出して音を制御する．
        """
        was_firing = self.states.get(thruster_id, False)

        # 噴射開始
        if is_firing and not was_firing:
            channel = pygame.mixer.find_channel()
            if channel:
                channel.play(self.sound_loop, loops=-1) # loops=-1で無限ループ
                self.channels[thruster_id] = channel
        # 噴射停止
        elif not is_firing and was_firing:
            # 継続音を停止
            if self.channels.get(thruster_id, None): # キーが存在しない場合も，キーが存在してもバリューが存在しない場合もある．
                self.channels[thruster_id].stop()
                self.channels[thruster_id] = None
            
            # 収束音
            shutoff_channel = pygame.mixer.find_channel()
            if shutoff_channel:
                shutoff_channel.play(self.sound_shutoff)
        
        # 状態を更新
        self.states[thruster_id] = is_firing