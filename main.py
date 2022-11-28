from control import ThymioControl
import time

# PROX_THRESHOLD = 2000 # ~4.5cm

# init path, position etc.
# while not at goal:
#   path following

# call sensors every 0.1? seconds
# if np.count_nonzero(thymio.get_prox() > PROX_THRESHOLD) > 0:
#     turn off repeated timer
#     while sensors detect something or not on path:
#         if obstacle: avoid
#         else: go to path
#     turn on repeated timer

thymio = ThymioControl()
thymio.path = [(100,100), (0,0)]
thymio.move_timer.start()

# del(thymio)