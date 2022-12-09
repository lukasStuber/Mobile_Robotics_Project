import matplotlib.pyplot as plt
import cv2
import math
import time
import numpy as np
from constant import DISTANCE_WHEELS, CALIBRATE_FACTOR, D_REAL, ROBOT_LENGTH

class utilsRobot:
    def __init__(self, thymio=None):
        self.th = thymio
        self.angle=0
        self.coords=(0,0)
        self.speed=(0,0)
        self.center_red=None
        self.center_green=None
        self.time_stamp=time.time()
        self.scaling_factor=1
        self.scaled_robot_length=self.scaling_factor*ROBOT_LENGTH
        self.stop()

    def get_current_speed(self):
        return self.th['motor.left.target']*CALIBRATE_FACTOR/self.scaling_factor, self.th['motor.right.target']*CALIBRATE_FACTOR/self.scaling_factor

    def move(self, l_speed, r_speed, verbose=False):
        if verbose: print("\t\t Setting speed : ", l_speed, r_speed)
        l_speed = l_speed if l_speed>=0 else 2**16+l_speed
        r_speed = r_speed if r_speed>=0 else 2**16+r_speed
        self.time_stamp=time.time()
        self.done_moving=False
        self.th.set_var('motor.left.target', l_speed)
        self.th.set_var('motor.right.target', r_speed)

    def stop(self):
        self.move(l_speed=0,r_speed=0)
        self.done_moving=True

    def go_straight(self, distance, speed):
        self.move(speed, speed)
        return distance/self.scaling_factor/(speed*CALIBRATE_FACTOR)

    def turn_angle(self, angle, speed):
        direction=1
        if angle<0: direction=-1
        self.move(-direction*speed, direction*speed)
        return abs(angle*DISTANCE_WHEELS/(2*CALIBRATE_FACTOR*speed)), direction

    def find_robot(self, image):
        def find_center_color(img, lower, upper, color,  blur_factor=11):
            blur_image=cv2.bilateralFilter(img,20,40,10)
            hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            _, mask = cv2.threshold(cv2.blur(mask, (blur_factor, blur_factor)),50,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(np.uint8(mask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            middles = []
            for c in contours:
                M = cv2.moments(c)
                if M["m00"]==0:
                    print("WARNING: Error with find one corners center"); continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                middles.append((cX, cY))
            if len(middles)>1:
                print("WARNING: Multiple found, doing average")
                for i in middles: cv2.circle(hsv, i, 4, (255,0,0), -1)
                plt.imshow(hsv); plt.show()
                return (round(sum([i[0] for i in middles])/len(middles)),round(sum([i[1] for i in middles])/len(middles))), mask
            elif len(middles)==0:
                print("ERROR: No middle found", color)
                plt.imshow(hsv); plt.show()
                return False, mask
            return middles[0], mask
        img=image.copy()
        lower_green = np.array([65 , 70, 40])
        upper_green = np.array([100, 155, 105])
        lower_blue = np.array([98 , 130, 60])
        upper_blue= np.array([115, 190, 150])
        center_green, green_mask=find_center_color(img, lower_green, upper_green, "green")
        center_red, red_mask=find_center_color(img, lower_blue, upper_blue, "blue")
        success=True
        if center_green: self.center_green=center_green
        else:success=False
        if center_red: self.center_red=center_red
        else:success=False
        angle = math.atan2((self.center_red[1]-self.center_green[1]),(self.center_red[0]-self.center_green[0]))+math.pi/2
        if angle>math.pi: angle-=2*math.pi
        green_mask=cv2.bitwise_not(cv2.dilate(green_mask, np.ones((4,4),np.uint8)))
        img = cv2.bitwise_and(cv2.bitwise_not(img), cv2.bitwise_not(img), mask = green_mask)
        red_mask=cv2.bitwise_not(cv2.dilate(red_mask, np.ones((4,4),np.uint8)))
        img = cv2.bitwise_not(cv2.bitwise_and(img, img, mask = red_mask))
        return success, (round((self.center_red[0]+self.center_green[0])/2),round((self.center_red[1]+self.center_green[1])/2)), angle, img

    def find_delta_angle(self, robot_angle, goal_angle):
        delta_angle=robot_angle-goal_angle
        while delta_angle>math.pi: delta_angle-=math.pi*2
        while delta_angle<-math.pi: delta_angle+=math.pi*2
        return delta_angle

    def image_pixel_scale(self):
        d = math.sqrt((self.center_red[0]-self.center_green[0])**2+(self.center_red[1]-self.center_green[1])**2)
        self.scaling_factor = d / D_REAL #px/mm
        self.scaled_robot_length=ROBOT_LENGTH*self.scaling_factor
        return self.scaling_factor

    def increment_motor(self, side, val):
        if side=="left":
            new_val=self.th['motor.left.target']+val
            if new_val<self.th['motor.right.target']:
                self.th.set_var('motor.left.target', new_val)
        elif side=="right":
            new_val=self.th['motor.right.target']+val
            if new_val<self.th['motor.left.target']:
                self.th.set_var('motor.right.target', new_val)