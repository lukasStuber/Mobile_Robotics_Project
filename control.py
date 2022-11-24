import math
import RepeatedTimer
from tdmclient import ClientAsync, aw

ODOMETRY_INTERVAL = 0.1
WHEEL_DISTANCE = 95
SPEED_TO_MMS = 0.325 # moves 65mm in 2s at 100 speed

class ThymioControl:
    def __init__(self, position=(0,0), angle=0):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        print(self.node)
        self.position = position
        self.angle = angle
        self.speed_target = (0,0)
        self.odometry_timer = RepeatedTimer.RepeatedTimer(ODOMETRY_INTERVAL, self.estimate_position)
        self.odometry_timer.start()

    def __del__(self):
        def unlock(self):
            aw(self.node.unlock())
        self.odometry_timer.cancel()
        self.client.close()

    def sleep(self, time):
        aw(self.client.sleep(time))

    def move(self, l_speed, r_speed=None):
        if r_speed is None: r_speed = l_speed
        aw(self.node.set_variables({"motor.left.target": [l_speed/SPEED_TO_MMS],"motor.right.target": [r_speed/SPEED_TO_MMS]}))
        self.speed_target = (l_speed, r_speed)
    
    # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
    def estimate_position(self):
        dx = ODOMETRY_INTERVAL*self.speed_target[0] # read motor speed?
        dy = ODOMETRY_INTERVAL*self.speed_target[1]
        da = (dx - dy)/WHEEL_DISTANCE
        dc = (dx + dy)/2
        self.position = (self.position[0] + dc*math.cos(self.angle), self.position[1] + dc*math.sin(self.angle))
        self.angle = self.angle + da
        print(self.position, self.angle)

    def correct_position(self, position, angle): # value from kalman filter
        self.position = position
        self.angle = angle