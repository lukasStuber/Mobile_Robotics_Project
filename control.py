import math, time
import numpy as np
from tdmclient import ClientAsync, aw
from threading import Timer
from RepeatedTimer import RepeatedTimer
from constants import *

class ThymioControl:
    def __init__(self, position=(0,0), angle=0, kalman=None):
        self.kalman = kalman
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        aw(self.node.set_variables({"leds.temperature": [int(0),int(0)]})) # "leds.circle": [0,0,0,0,0,0,0,0], "leds.rc": [0], 
        self.position = position
        self.angle = angle
    # path following
        self.stop_planned = False
        self.path = [(0,0)]
        self.path_index = 0
    # obstacle avoidance
        self.proxs = np.zeros(5)
        self.obst_direction = 0
    # odometry
        self.odometry_time = None
        self.speed_target = (0,0)
    # timers
        self.stop_timer = Timer(0, self.stop)
        self.move_timer = RepeatedTimer(MOVE_INTERVAL, self.navigation)
        self.odometry_timer = RepeatedTimer(ODOMETRY_INTERVAL, self.estimate_position)

# MOVEMENT
    def set_coordinates(self, position, angle):
        self.position = position
        self.angle = angle

    def move(self, l_speed, r_speed=None):
        if r_speed is None: r_speed = l_speed
        aw(self.node.set_variables({"motor.left.target": [int(l_speed)],"motor.right.target": [int(r_speed)]}))
        self.speed_target = (l_speed, r_speed)

    def stop(self):
        self.stop_planned = False
        self.move(0)

    def timed_move(self, l_speed, r_speed, time):
        self.stop_planned = True
        self.stop_timer = Timer(time, self.stop)
        self.move(l_speed, r_speed)
        self.stop_timer.start()

    def navigation(self):
        self.get_prox()
        if np.any(self.proxs > PROX_THRESHOLD): self.avoid_obstacles()
        else: self.move_to_goal()

    def follow_path(self):
        self.odometry_timer.start()
        self.move_timer.start()

# PATH FOLLOWING
    def move_to_goal(self):
        print('moving')
        if self.stop_planned: return
        if self.obst_direction != 0: # avoid previously detected obstacle
            self.timed_move(-self.obst_direction*OBST_TURN_SPEED + OBST_SPEED, self.obst_direction*OBST_TURN_SPEED + OBST_SPEED, OBST_TIME)
            self.obst_direction = 0
            return
        goal = self.path[self.path_index]
        dist = math.sqrt((goal[0] - self.position[0])**2 + (goal[1] - self.position[1])**2)
        if dist > DIST_TOL:
            angle = (math.atan2(goal[1] - self.position[1], goal[0] - self.position[0]) - self.angle + math.pi) % (2*math.pi) - math.pi
            if abs(angle) > ANGLE_TOL:
                direction = 1 if angle > 0 else -1 # 1 = turn left, -1 = turn right
                t = abs(angle)*WHEEL_DIST / (2*STANDARD_SPEED*SPEED_TO_MMS) - 0.1 # remove the await delay of move()
                t = min(t, MAX_TIME) # clamp to allow for obstacle avoidance
                self.timed_move(-direction*STANDARD_SPEED, direction*STANDARD_SPEED, t) # let kalman correct halfway
            else:
                t = dist / (STANDARD_SPEED*SPEED_TO_MMS) - 0.1 # remove the await delay of move()
                t = min(t, MAX_TIME) # clamp to allow for obstacle avoidance
                self.timed_move(STANDARD_SPEED, STANDARD_SPEED, t) # let kalman correct halfway
        else:
            self.path_index += 1
            if self.path_index >= len(self.path): self.end_path()
    
    def set_path(self, path):
        self.path = path
        self.path_index = 0

    def end_path(self):
        self.move_timer.stop()
        self.odometry_timer.stop()
        self.stop_timer.cancel()
        self.stop()
        #self.crab_rave()
    
# OBSTACLE AVOIDANCE
    def get_prox(self):
        aw(self.node.wait_for_variables({"prox.horizontal"}))
        print(list(self.node.v.prox.horizontal))
        self.proxs = np.array(list(self.node.v.prox.horizontal))[:-2]

    def avoid_obstacles(self):
        print('avoiding')
        self.stop_planned = False
        self.stop_timer.cancel()
        speed = np.dot(self.proxs, [[1, -1], [3, -3], [-3, 3], [-3, 3], [-1, 1]])/100
        if ((self.proxs[2]+self.proxs[1]+self.proxs[3])<30):
           speed[0] += 100; speed[1] += 100
        self.obst_direction = 1 if speed[0] > speed[1] else -1 # turn left when obstacle on the left and vice versa
        self.move(speed[0], speed[1])

# ODOMETRY
    def estimate_position(self):
        if self.odometry_time is None: # initialisation
            self.odometry_time = time.time()
            return
        interval = time.time() - self.odometry_time
        self.odometry_time = time.time()
        # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
        dx = interval*SPEED_TO_MMS*self.speed_target[0] # fetching the actual speed from the thymio is very slow (100ms) and inconsistent
        dy = interval*SPEED_TO_MMS*self.speed_target[1] # odometry needs a high frequency to be accurate
        da = (dy - dx)/WHEEL_DIST
        dc = (dx + dy)/2
        self.position = (self.position[0] + dc*math.cos(self.angle), self.position[1] + dc*math.sin(self.angle))
        self.angle = (self.angle + da) % (2*math.pi)
        self.kalman.state_prop(self.speed_target)
        print(self.position, self.angle*180/math.pi)

    def get_estimate(self):
        return (self.position[0]*0.001, self.position[1]*0.001, self.angle) # in meters for kalman filter
    
    def get_speed(self):
        return SPEED_TO_MMS*self.speed_target # in mm/s for kalman filter

# OTHER :)))
    def crab_rave(self):
        program = '''
var note[19] = [2349, 1976, 1568, 1568, 2349, 2349, 1760, 1397, 1397, 2349, 2349, 1760, 1397, 1397, 2094, 2094, 1319, 1319, 1397]
var duration[19] = [2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1]
var i = 1
var play = 1
call sound.freq(note[0]/2, 8*duration[0])
onevent sound.finished
    if play == 1 then
        call sound.freq(note[i]/2, 8*duration[i])
        i = i + 1
        if i == 19 then
            i = 0
        else
        end
    end
onevent button.center
    play = 0
        '''
        async def prog():
            await self.node.compile(program)
            await self.node.run()
        self.client.run_async_program(prog)