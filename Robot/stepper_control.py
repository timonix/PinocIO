from threading import Thread
from time import sleep
import RPi.GPIO as GPIO
from enum import Enum


class Steppers:

    next_action = ''
    action_active = False

    m1_en = 24
    m1_step = 14
    m1_dir = 15

    action = Enum('action', 'FORWARD BACKWARD TURN_LEFT TURN_RIGHT ABORT')

    t_check_action = None

    # success_flag = False        # Used to make the other processes aware of complete action?

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.m1_en, GPIO.OUT)
        GPIO.setup(self.m1_step, GPIO.OUT)
        GPIO.setup(self.m1_dir, GPIO.OUT)

        GPIO.output(self.m1_en, GPIO.LOW)
        GPIO.output(self.m1_step, GPIO.LOW)
        GPIO.output(self.m1_dir, GPIO.LOW)

    def go_forward(self):

        self.action_active = True

        for i in range(16000):
            GPIO.output(self.m1_step, GPIO.HIGH)
            sleep(0.0001)
            GPIO.output(self.m1_step, GPIO.LOW)
            sleep(0.0001)

        self.action_active = False

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

    def check_action(self):  # Made to loop in a separate thread to monitor action and send to movement controller
        print("Starting check input loop")

        while True:

            if self.action_active is False and self.next_action is not '':
                print("Doing action")
                if self.next_action == self.action.FORWARD:
                    t = Thread(target=self.go_forward)
                elif self.next_action == self.action.BACKWARD:
                    t = Thread(target=self.go_backward)
                elif self.next_action == self.action.TURN_LEFT:
                    t = Thread(target=self.turn_left)
                elif self.next_action == self.action.TURN_RIGHT:
                    t = Thread(target=self.turn_right)
                else:
                    print("Action not allowed. Breaking loop")

                t.start()

            if self.next_action == self.action.ABORT:
                break

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



