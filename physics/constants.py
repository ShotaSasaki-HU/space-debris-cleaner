# physics/constants.py
"""
物理シミュレーション用の定数モジュール
SI単位系とCanonical Units（DU, MU, TU）の変換定数を提供
"""

# 地球のパラメータ（SI単位系ベース）
EARTH_RADIUS_M = 6378.137e3
EARTH_MASS_KG = 5.972e24
G_CONSTANT_SI = 6.67430e-11 # m^3 / (kg * s^2)

# カノニカル単位系への変換係数
METER_TO_DU = 1.0 / EARTH_RADIUS_M
KG_TO_MU = 1.0 / EARTH_MASS_KG

# カノニカル単位系において，GM = 1 となるような時間の単位（Time Unit）を計算．
# なぜなら，重力の式をめちゃシンプルにしたいから．
TU_TO_SEC = (EARTH_RADIUS_M ** 3 / (G_CONSTANT_SI * EARTH_MASS_KG)) ** 0.5
SEC_TO_TU = 1.0 / TU_TO_SEC

# カノニカル空間における万有引力定数
# 定義により1.0になる．万有引力による加速度を，SI単位系とカノニカル単位系で比較するとわかるよ．
G_CANONICAL = 1.0

# print(METER_TO_DU * 6378100) # 1 [CDU] = 6378.1 km = 20,925,524.97 ft
# print(METER_TO_DU / SEC_TO_TU * 7905.38) # 1 [CDU]/[CTU] = 7.90538 km/s = 25,936.29 ft/sec
# print(SEC_TO_TU * 806.80415) # 1 [CTU] = 806.80415 s
