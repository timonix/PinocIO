from threading import Thread
from time import sleep
import RPi.GPIO as GPIO
from enum import Enum
from gpiozero import Servo
from time import sleep

servo = Servo(17)

while True:
    try:
        servo.value = -1
        sleep(2)

        for x in range(-1, 1, 0.1):
            servo.value = -1
            sleep(0.5)

        sleep(2)
    except KeyboardInterrupt:
        print("Program stopped")
        GPIO.cleanup()
        exit()

class RobotControl:

    next_action = ''
    action_active = False

    m1_en = 24
    m1_step = 14
    m1_dir = 15

    action = Enum('action', 'FORWARD BACKWARD TURN_LEFT TURN_RIGHT ABORT')

    action_loop_thread = None

    def __init__(self):

        servo = Servo(17)

        self.action_loop_thread = Thread(target=self.check_action)
        self.action_loop_thread.start()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.m1_en, GPIO.OUT)
        GPIO.setup(self.m1_step, GPIO.OUT)
        GPIO.setup(self.m1_dir, GPIO.OUT)

        GPIO.output(self.m1_en, GPIO.LOW)
        GPIO.output(self.m1_step, GPIO.LOW)
        GPIO.output(self.m1_dir, GPIO.LOW)

    def stepper_control(self, motor, steps):
        pass    # TODO

    def angle_to_steps_calculation(self, angle):
        pass    # TODO

    def go_forward(self):
        for i in range(200):
            GPIO.output(self.m1_step, GPIO.HIGH)
            sleep(0.0001)
            GPIO.output(self.m1_step, GPIO.LOW)
            sleep(0.0001)

    def go_backward(self):
        print("Backward START")
        sleep(3)
        print("Backward STOP")

    def turn_right(self):
        print("Turning right START")
        sleep(3)
        print("Turning right STOP")

    def turn_left(self):
        print("Turning left START")
        sleep(3)
        print("Turning left STOP")

    def check_action(self):  # Made to loop in a separate thread to monitor and perform actions
        print("Starting stepper control loop thread.")

        while True:

            if self.next_action == self.action.ABORT:
                print("Aborting stepper control loop thread.")
                break

            if self.next_action == self.action.FORWARD:
                self.go_forward()
            elif self.next_action == self.action.BACKWARD:
                self.go_backward()
            elif self.next_action == self.action.TURN_LEFT:
                self.turn_left()
            elif self.next_action == self.action.TURN_RIGHT:
                self.turn_right()

    def do_movement(self, movement):

        self.action_active = True
        print("before movement")

        if movement == 'forward':
            t = Thread(target=self.go_forward)
            t.start()
        elif movement == 'backward':
            t = Thread(target=self.go_backward)
            t.start()
        elif movement == 'right':
            t = Thread(target=self.turn_right)
            t.start()
        elif movement == 'left':
            t = Thread(target=self.turn_left)
            t.start()

        self.action_active = False
        print("after movement")



