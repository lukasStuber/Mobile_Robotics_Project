import math
import numpy as np
from threading import Timer
from RepeatedTimer import RepeatedTimer
from tdmclient import ClientAsync, aw
from constants import *

class ThymioControl:
    def __init__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        self.position = (0,0)
        self.angle = 0
        self.speed_target = (0,0)
    # path following
        self.stop_planned = False
        self.path = [(0,0)]
        self.path_index = 0
    # obstacle avoidance
        self.proxs = np.zeros(5)
        self.obst_direction = 0
    # timers
        self.stop_timer = Timer(0, self.stop)
        self.move_timer = RepeatedTimer(MOVE_INTERVAL, self.navigation)

# MOVEMENT
    def move(self, l_speed, r_speed=None):
        if r_speed is None: r_speed = l_speed
        aw(self.node.set_variables({"motor.left.target": [int(l_speed)],
                    "motor.right.target": [int(r_speed)]}))
        self.speed_target = (l_speed, r_speed)

    def stop(self):
        self.stop_planned = False
        self.move(0)

    def timed_move(self, l_speed, r_speed, time):
        self.stop_planned = True
        self.stop_timer = Timer(time - 0.1, self.stop)  # compensate stopping delay
        self.move(l_speed, r_speed)
        self.stop_timer.start()

    def navigation(self):
        self.get_prox()
        if np.any(self.proxs > PROX_THRESHOLD):
            print(self.proxs); self.avoid_obstacles()
        else: self.move_to_goal()

    def follow_path(self):
        self.move_timer.start()

# PATH FOLLOWING
    def move_to_goal(self):
        if self.stop_planned: return
        if self.obst_direction != 0: # avoid previously detected obstacle
            self.timed_move(self.obst_direction*OBST_TURN_SPEED + OBST_SPEED,
                            -self.obst_direction*OBST_TURN_SPEED + OBST_SPEED, OBST_TIME)
            self.obst_direction = 0
            return
        goal = self.path[self.path_index]
        dist = math.sqrt((goal[0] - self.position[0])**2 + (goal[1] - self.position[1])**2)
        if dist > DIST_TOL:
            angle = (math.atan2(goal[1] - self.position[1], goal[0] - self.position[0])
                        - self.angle + math.pi) % (2*math.pi) - math.pi
            if abs(angle) > ANGLE_TOL:
                direction = 1 if angle > 0 else -1 # 1 = turn left, -1 = turn right
                t = abs(angle)*WHEEL_DIST/(2*STANDARD_SPEED*SPEED_TO_MMS)
                t = min(t, MAX_TIME) # allow kalman updates
                self.timed_move(direction*STANDARD_SPEED, -direction*STANDARD_SPEED, t)
            else:
                t = dist/(STANDARD_SPEED*SPEED_TO_MMS)
                t = min(t, MAX_TIME) # allow kalman updates
                self.timed_move(STANDARD_SPEED, STANDARD_SPEED, t)
        else:
            self.path_index += 1
            if self.path_index >= len(self.path): self.end_path()
    
    def set_path(self, path):
        self.path = path
        self.path_index = 0

    def end_path(self):
        self.move_timer.stop()
        self.stop_timer.cancel()
        self.stop()
        self.crab_rave()
    
# OBSTACLE AVOIDANCE
    def get_prox(self):
        aw(self.node.wait_for_variables({"prox.horizontal"}))
        self.proxs = np.array(list(self.node.v.prox.horizontal))[:-2]

    def avoid_obstacles(self):
        self.stop_planned = False
        self.stop_timer.cancel()
        speed = np.dot(self.proxs, [[3, -3], [1, -1], [-1, -1], [-1, 1], [-3, 3]])/100
        if (np.sum(self.proxs[1:4]) < 30): # if no obstacle in front, move forward
           speed[0] += STANDARD_SPEED/2; speed[1] += STANDARD_SPEED/2
        self.obst_direction = 1 if speed[0] < speed[1] else -1 # go left when obstacle on left and vice versa
        self.move(speed[0], speed[1])

# OTHER
    def keyboard(self):
        import keyboard
        while True:
            if keyboard.is_pressed('q'):
                thymio.stop(); break
            vl = 0; vr = 0
            if keyboard.is_pressed('w'):
                vl += 250; vr += 250
            if keyboard.is_pressed('s'):
                vl -= 250; vr -= 250
            if keyboard.is_pressed('a'):
                vl -= 250; vr += 250
            if keyboard.is_pressed('d'):
                vl += 250; vr -= 250
            thymio.move(vl, vr)

    def crab_rave(self):
        program = '''
var note[19] = [2349, 1976, 1568, 1568, 2349, 2349, 1760, 1397, 1397, 2349, 2349, 1760, 1397, 1397, 2094, 2094, 1319, 1319, 1397]
var duration[19] = [2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2]
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

if __name__ == '__main__':
    from Kalman import Kalman
    kalman = Kalman(); thymio = ThymioControl()
    def odometry():
        kalman.state_prop(thymio.speed_target)
        thymio.position[0] = kalman.x[0,0]
        thymio.position[1] = kalman.x[1,0]
        thymio.angle = kalman.x[2,0]
        # print(thymio.position[0], thymio.position[1], thymio.angle*180/math.pi)
    odometry_timer = RepeatedTimer(ODOMETRY_INTERVAL, odometry)
    odometry_timer.start()
    thymio.set_path([(0,0), (100,0), (100,100), (0,100), (0,0)])
    thymio.follow_path()
    # thymio.keyboard()