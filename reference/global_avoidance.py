import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from constant import ERODE_FACTOR, ERODE_FACTOR_CONNECTION_RATIO

class Global_Avoidance():
    def __init__(self):
        self.map=None               #picture of the plane
        self.patched_map=None       #picture of the plane, with white spots on robot and goal
        self.obstacle_map=None      #picture of obstacles, filtered and processed
        self.start_coords=(0,0)     #coordinates of starting point
        self.goal_coords=(0,0)      #coordinates of goal point
        self.corners=None           #list of obstacles corners detected
        self.connections=None       #list of possible connections between corners
        self.path=None              #shortest path

    def find_obstacles(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.bilateralFilter(img,20,75,500)
        _, self.obstacle_map = cv2.threshold(filtered_img,100,255,cv2.THRESH_BINARY)

    def find_path(self):
        def find_corners(gray_layout, ERODE_FACTOR=30):
            '''
            The corners here are computed the Harris algorithm. This function is inspired by
            https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html
            Inputs: - Obstacle map : 'gray_layout' is a processed version of the obstacle map computed from the function 'find_obstacles'.
                    - Erode factor : value describing the safety margin used to make sure the computation for the possible paths takes into account the width of the robot.
            Output: - Corners : array of (x,y) coordinates of the corners of the obstacles detected.
            '''
            kernel = np.ones((ERODE_FACTOR,ERODE_FACTOR),np.uint8)
            gray_eroded = cv2.erode(gray_layout,kernel,iterations = 1)
            corner_harris = cv2.cornerHarris(gray_eroded,10,3,0.04)
            _, corner_harris = cv2.threshold(corner_harris,0.01*corner_harris.max(),1,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(np.uint8(corner_harris),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #find contours of corners
            corners = [self.start_coords, self.goal_coords]
            for c in contours:
                M = cv2.moments(c)
                if M["m00"]==0:
                    print("WARNING: Error with find one corners center"); continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                corners.append((cX, cY))
            return corners

        def connect_corners(corners, layout, ERODE_FACTOR):
            kernel = np.ones((round(ERODE_FACTOR*ERODE_FACTOR_CONNECTION_RATIO),round(ERODE_FACTOR*ERODE_FACTOR_CONNECTION_RATIO)),np.uint8)
            layout = cv2.erode(layout,kernel,iterations = 1)
            connections={}
            for i in range(len(corners)):
                for j in range(i+1, len(corners)):
                    x0, x1, y0, y1=corners[i][0], corners[j][0], corners[i][1], corners[j][1]
                    connection_possible=True
                    if abs(y1-y0)>abs(x1-x0):               #for angle more than 45°, we use y as increment
                        if y0>y1: y1,y0,x1,x0=y0,y1,x0,x1
                        a=(x1-x0)/(y1-y0)
                        b=x0-a*y0
                        for y in range(y0,y1):
                            if layout[y][int(a*y+b)]<200:   #using gray as it is gary scal and not RGB
                                connection_possible=False; break
                    elif abs(x1-x0)>abs(y1-y0):             #otherwise, x
                        if x0>x1: x1,x0,y1,y0=x0,x1,y0,y1
                        a=(y1-y0)/(x1-x0)
                        b=y0-a*x0
                        for x in range(x0,x1):
                            if layout[int(a*x+b)][x]<200:   #using gray as it is gary scal and not RGB
                                connection_possible=False; break
                    else: continue
                    if connection_possible:
                        length=((x0-x1)**2+(y0-y1)**2)**0.5
                        if not(corners[i] in connections): connections[corners[i]]={}
                        connections[corners[i]][corners[j]]=length
                        if not(corners[j] in connections): connections[corners[j]]={}
                        connections[corners[j]][corners[i]]=length
            return connections

        def dijkstra(connections, start, goal):
            nodes_to_visit=connections.copy()
            nearest={}; predecessor={}
            for node in nodes_to_visit: nearest[node]=math.inf
            nearest[start]=0
            while(nodes_to_visit):
                min_Node=None
                for node in nodes_to_visit:
                    if min_Node==None: min_Node=node
                    elif nearest[min_Node] > nearest[node]: min_Node=node
                for child_node, value in nodes_to_visit[min_Node].items():
                    if value+nearest[min_Node] < nearest[child_node]:
                        nearest[child_node] = value + nearest[min_Node]
                        predecessor[child_node] = min_Node
                nodes_to_visit.pop(min_Node)
            path=[]
            node = goal
            while node != start:
                try:
                    path.insert(0,node); node = predecessor[node]
                except Exception:
                    print('Path not reachable'); return None
            path.insert(0,start)
            if nearest[goal] != math.inf: return path
            else:
                print('Path not reachable'); return None
        gray = np.float32(self.obstacle_map)
        if gray[self.start_coords[1]][self.start_coords[0]]==0 or gray[self.goal_coords[1]][self.goal_coords[0]]==0:
            print("ERROR: Starting point or ending point on obstacle"); return False
        #find corners using Harris algorithm, inspired from https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html
        self.corners=find_corners(gray, ERODE_FACTOR=ERODE_FACTOR)
        #connect corners if possible
        self.connections=connect_corners(self.corners, gray, ERODE_FACTOR=ERODE_FACTOR)
        #find path using dijkstra inpired from https://www.pythonpool.com/dijkstras-algorithm-python/
        self.path=dijkstra(self.connections, self.start_coords, self.goal_coords)
        if self.path==None:
            self.display_path(self.map, freeze=True)
            print("ERROR: No path found"); return False
        return True

    def display_path(self, image, robot_coords=False, est_robot_coords=False, freeze=False, path=True, corners=True, connections=True):
        img=image.copy()
        if path:
            if connections:
                for i in self.connections:
                    for j in self.connections[i]: cv2.line(img, i, j, (0,255,255), 2)
            if corners:
                for i in self.corners: cv2.circle(img, i, 3, (0,150,0), -1)
            if self.path:
                for i in range(len(self.path)-1): cv2.line(img, self.path[i], self.path[i+1], (0,0,255), 3)
        cv2.circle(img, self.start_coords, 4, (0,0,255), -1)
        cv2.circle(img, self.goal_coords, 4, (255,0,0), -1)
        if robot_coords:
            P=(round(robot_coords[0]), round(robot_coords[1]))
            cv2.circle(img, P, 4, (255,0,0), -1)
        if est_robot_coords:
            P=(round(est_robot_coords[0]), round(est_robot_coords[1]))
            cv2.circle(img, P, 4, (0,0,255), -1)
        plt.imshow(img)
        if freeze: plt.show()
        else:
            plt.draw(); plt.pause(0.0001); plt.clf()

    def find_goal(self, image):
        def find_center_color(img, lower, upper, color,  blur_factor=11):
            blur_image=cv2.bilateralFilter(img,20,40,10)
            hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.blur(mask, (blur_factor, blur_factor))
            _, mask = cv2.threshold(mask,50,255,cv2.THRESH_BINARY)
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
                print("Multiple found, doing average")
                return (round(sum([i[0] for i in middles])/len(middles)),round(sum([i[1] for i in middles])/len(middles))), mask
            elif len(middles)==0:
                print("No middle found", color)
                plt.imshow(hsv); plt.show()
                return False, mask
            return middles[0], mask
        img=image.copy()
        lower_red = np.array([165 , 90, 120])
        upper_red = np.array([185, 190, 230])
        self.goal_coords,mask=find_center_color(img, lower_red, upper_red, "red")
        if self.goal_coords==False: return False
        mask=cv2.bitwise_not(cv2.dilate(mask, np.ones((4,4),np.uint8)))
        self.patched_map =cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(img), cv2.bitwise_not(img), mask = mask))
        return True