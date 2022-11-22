import numpy as np
import math
import time
from constant import EXT_SPEED, INT_SPEED, LA_MOVE_TIME

class Local_Avoidance:
    def __init__(self, thymio, uR):
        self.th = thymio
        self.ur = uR
        self.side=None

    def sawWall(self, threshold=500):
        return any([x>threshold for x in self.th['prox.horizontal'][:-2]])

    def sawBlack(self, threshold = 200):
        return any([x<=threshold for x in self.th['prox.ground.reflected']])

    def avoid_obstacle(self, rob_speed, new_state, motion_callback, increment_callback):
        if not self.sawBlack():
            while self.sawWall():
                capt = np.array(self.th["prox.horizontal"][:-2])
                ratio = np.array([[1, -1], [1/2, -1/2], [-1/4, 1/4], [-1/2, 1/2], [-1, 1]])
                speed = np.dot(capt,ratio)
                speed = (1/10)*speed + rob_speed
                if speed[0] >= speed[1]: new_state="right"
                else: new_state="left"
                self.ur.move(int(speed[0]), int(speed[1]))
            time.sleep(0.5)
            self.ur.stop()
        else:
            pass
        if new_state=="left" or new_state=="right":
            if new_state=="left":
                left_speed=EXT_SPEED; right_speed=INT_SPEED
                self.side="left"
            elif new_state=="right":
                left_speed=INT_SPEED; right_speed=EXT_SPEED
                self.side="right"
            motion_callback.interval=LA_MOVE_TIME
            self.ur.move(left_speed,right_speed)
            motion_callback.start(); increment_callback.start()