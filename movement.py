import math
import RepeatedTimer
import time
from Thymio import Thymio

# TODO
# speed-values to mm conversion

ODOMETRY_INTERVAL = 0.1
WHEEL_DISTANCE = 95

class Movement:
    def __init__(self, position=(0,0), angle=0, thymio=None):
        self.position = position
        self.angle = angle
        self.thymio = thymio
        self.speed = (0,0)
        self.odometry_timer = RepeatedTimer.RepeatedTimer(ODOMETRY_INTERVAL, self.estimate_position)
        self.odometry_timer.start()

    def move(self, speed):
        speed = speed if speed >= 0 else speed + 2**16
        self.thymio.set_var('motor.left.target', speed)
        self.thymio.set_var('motor.right.target', speed)
        self.speed = (speed, speed)

    def move(self, l_speed, r_speed):
        l_speed = l_speed if l_speed >= 0 else l_speed + 2**16
        r_speed = r_speed if r_speed >= 0 else r_speed + 2**16
        self.thymio.set_var('motor.left.target', l_speed)
        self.thymio.set_var('motor.right.target', r_speed)
        self.speed = (l_speed, r_speed)

    def turn(self, angle, speed):
        direction = copysign(1, angle)
        self.move(-direction*speed, direction*speed)
    
    # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
    def estimate_position(self):
        dx = ODOMETRY_INTERVAL*self.speed[0]
        dy = ODOMETRY_INTERVAL*self.speed[1]
        da = (dx - dy)/WHEEL_DISTANCE
        dc = (dx + dy)/2
        self.position = (self.position[0] + dc*math.cos(self.angle), self.position[1] + dc*math.sin(self.angle))
        self.angle = self.angle + da
        #print(self.position, self.angle)

    def correct_position(self, position, angle):
        self.position = position
        self.angle = angle