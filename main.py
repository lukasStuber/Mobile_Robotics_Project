from control import ThymioControl
import time

thymio = ThymioControl()
thymio.move(100)
time.sleep(2)
thymio.move(0,0)
thymio.unlock()