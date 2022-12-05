from control import ThymioControl
# import time

# image processing
# pathfinding
# start kalman, which calls image processing
# start path following
thymio = ThymioControl(position=(0,0), angle=0)
path = [(500,0)]
thymio.set_path(path)
thymio.follow_path()