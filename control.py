import math
import RepeatedTimer
from tdmclient import ClientAsync, aw

ODOMETRY_INTERVAL = 0.1
WHEEL_DISTANCE = 95
MMS_SPEED = 0.325 # to get mm/s from speed; thymio moves 65mm in 2s at 100 speed

class ThymioControl:
    def __init__(self, position=(0,0), angle=0):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        # print(self.node)
        self.position = position
        self.angle = angle
        self.speed_target = (0,0)
        self.odometry_timer = RepeatedTimer.RepeatedTimer(ODOMETRY_INTERVAL, self.estimate_position)
        self.odometry_timer.start()

    def __del__(self):
        self.unlock()
        self.odometry_timer.cancel()
        self.client.close()

    def unlock(self):
        aw(self.node.unlock())

    def sleep(self, time): # might be better than time.sleep()? it's used in the library doc
        aw(self.client.sleep(time))

    def move(self, l_speed, r_speed=None): # speed in mm/s
        if r_speed is None: r_speed = l_speed
        aw(self.node.set_variables({"motor.left.target": [(int)(l_speed/MMS_SPEED)],"motor.right.target": [(int)(r_speed/MMS_SPEED)]}))
        # self.node.v.motor.left.target = l_speed/MMS_SPEED
        # self.node.v.motor.right.target = r_speed/MMS_SPEED
        # self.node.flush() # moves the changed variables to the thymio; probably slower than set_variables
        self.speed_target = (l_speed, r_speed)
        # print(self.speed_target)
    
    # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
    def estimate_position(self):
        aw(self.node.wait_for_variables({"motor.left.speed", "motor.right.speed"}))
        dx = ODOMETRY_INTERVAL*MMS_SPEED*self.node.v.motor.left.speed
        dy = ODOMETRY_INTERVAL*MMS_SPEED*self.node.v.motor.right.speed
        # dx = ODOMETRY_INTERVAL*self.speed_target[0] # in case the readings are weird or the conversion value is wrong
        # dy = ODOMETRY_INTERVAL*self.speed_target[1]
        da = (dx - dy)/WHEEL_DISTANCE
        dc = (dx + dy)/2
        self.position = (self.position[0] + dc*math.cos(self.angle), self.position[1] + dc*math.sin(self.angle))
        self.angle = self.angle + da
        print(self.position, self.angle)

    def correct_position(self, position, angle): # correction from the kalman filter
        self.position = position
        self.angle = angle