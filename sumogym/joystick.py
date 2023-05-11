

import inputs
import time
import threading

devices = inputs.devices.gamepads


class Joystick():

    def __init__(self) -> None:
        self.left_axis_x = 0
        self.left_axis_y = 0
        self.right_axis_x = 0
        self.right_axis_y = 0
        x = threading.Thread(target=self.main, args=(1,), daemon=True)
        x.start()
        pass

    def main(self, arg):
        while True:
            time.sleep(0.005)
            events = inputs.get_gamepad()
            for event in events:
                if event.ev_type == "Absolute" and event.code.startswith("ABS_"):
                    axis = event.code.replace("ABS_", "")
                    value = event.state
                    if axis == "X":
                        self.left_axis_x = value/32768.0
                    if axis == "Y":
                        self.left_axis_y = -value/32768.0
                    if axis == "RX":
                        self.right_axis_x = value/32768.0                        
                    if axis == "RY":
                        self.right_axis_y = -value/32768.0

            # print([self.left_axis, self.right_axis])


if __name__ == "__main__":
    joy = Joystick()
    while True:
        print("test")
        time.sleep(0.001)
