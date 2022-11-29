import math, time
import numpy as np
from tdmclient import ClientAsync, aw
from threading import Timer
from RepeatedTimer import RepeatedTimer

SPEED_TO_MMS = 0.32
STANDARD_SPEED = 200 # 32cm/5s
WHEEL_DIST = 95 # mm
DIST_TOL = 10 # mm
ANGLE_TOL = 0.1 # rad
MOVE_INTERVAL = 0.15 # s
ODOMETRY_INTERVAL = 0.025 # s
PROX_INTERVAL = 0.1 # s
PROX_THRESHOLD = 2000 # ~4.5cm

class ThymioControl:
    def __init__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        self.position = (0,0)
        self.angle = 0
        # path following
        self.moving = False
        self.path = [(100,100)]
        self.path_index = 0
        # odometry
        self.odometry_time = None
        self.speed_target = [0,0]
        self.move_timer = RepeatedTimer(MOVE_INTERVAL, self.move_to_goal)
        self.odometry_timer = RepeatedTimer(ODOMETRY_INTERVAL, self.estimate_position)
        self.prox_timer = RepeatedTimer(PROX_INTERVAL, self.obstacle_avoidance)
        self.odometry_timer.start()

    def move(self, l_speed, r_speed=None):
        if r_speed is None: r_speed = l_speed
        aw(self.node.set_variables({"motor.left.target": [int(l_speed)],"motor.right.target": [int(r_speed)]}))
        self.speed_target = [l_speed, r_speed]

    def stop(self):
        self.move(0)
        self.moving = False

    def timed_move(self, l_speed, r_speed, time):
        self.moving = True
        stop_timer = Timer(time, self.stop)
        self.move(l_speed, r_speed)
        stop_timer.start()

    def move_to_goal(self):
        if self.moving: return
        goal = self.path[self.path_index]
        dist = math.sqrt((goal[0]-self.position[0])**2 + (goal[1]-self.position[1])**2)
        if dist > DIST_TOL:
            angle = (math.atan2(goal[1]-self.position[1], goal[0]-self.position[0]) - self.angle + math.pi) % (2*math.pi) - math.pi
            if abs(angle) > ANGLE_TOL:
                print("turning" + str(angle))
                direction = 1 if angle > 0 else -1
                t = abs(angle)*WHEEL_DIST / (2*STANDARD_SPEED*SPEED_TO_MMS) - 0.1
                self.timed_move(-direction*STANDARD_SPEED, direction*STANDARD_SPEED, t)
            else:
                print("moving" + str(dist))
                t = dist / (STANDARD_SPEED*SPEED_TO_MMS) - 0.1
                self.timed_move(STANDARD_SPEED, STANDARD_SPEED, t)
        else:
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.move_timer.stop()
                self.odometry_timer.stop()
                self.prox_timer.stop()
                self.stop()
                self.crab_rave()
    
    # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
    def estimate_position(self):
        if self.odometry_time is None:
            self.odometry_time = time.time()
            return
        interval = time.time() - self.odometry_time
        self.odometry_time = time.time()
        dx = interval*SPEED_TO_MMS*self.speed_target[0] # reading the bot speeds takes frgn 100ms
        dy = interval*SPEED_TO_MMS*self.speed_target[1] # so does writing to them, thus a remote PID is practically impossible
        da = (dy - dx)/WHEEL_DIST # and a local PID would be hell because the thymio doesn't know floats
        dc = (dx + dy)/2
        self.position = (self.position[0] + dc*math.cos(self.angle), self.position[1] + dc*math.sin(self.angle))
        self.angle = (self.angle + da) % (2*math.pi)
        if not self.moving: print(self.position, self.angle*180/math.pi)

    def get_estimate(self):
        return (self.position[0]*0.001, self.position[1]*0.001, self.angle)

    def get_prox(self):
        return np.array(aw(self.node.get_variables(["prox.horizontal"]))[:-2])
    
    def obstacle_avoidance(self):
        if np.any(self.get_prox() > PROX_THRESHOLD):
            self.stop()
            self.move.timer.stop()
            aw(self.node.set_variables({"leds_top": [32,0,0]}))
            while np.any(proxs > PROX_THRESHOLD):
                ratio = np.array([[1, -1], [1/2, -1/2], [-1/4, 1/4], [-1/2, 1/2], [-1, 1]])
                speed = np.dot(proxs, ratio)
                speed = (1/10)*speed
                self.ur.move(int(speed[0]), int(speed[1]))
                proxs = self.get_prox()        

    def __del__(self):
        self.move_timer.stop()
        self.odometry_timer.stop()
        self.stop()
        self.client.close()
    
    def crab_rave(self):
        program = '''
var note[19] = [2349, 1976, 1568, 1568, 2349, 2349, 1760, 1397, 1397, 2349, 2349, 1760, 1397, 1397, 2094, 2094, 1319, 1319, 1397]
var duration[19] = [2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1]
var i = 1
call sound.freq(note[0]/2, 8*duration[0])
onevent sound.finished
    call sound.freq(note[i]/2, 8*duration[i])
    i = i + 1
    if i == 19 then
        i = 0
    else
    end
        '''
        async def prog():
            error = await self.node.compile(program)
            if error is not None: print(f"Compilation error: {error['error_msg']}")
            else:
                error = await self.node.run()
                if error is not None: print(f"Error {error['error_code']}")
        self.client.run_async_program(prog)