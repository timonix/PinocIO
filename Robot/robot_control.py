from threading import Thread
from time import sleep
import RPi.GPIO as GPIO
from enum import Enum
from gpiozero import Servo
from time import sleep
import math


class RobotControl:

    # --- Class things ---

    action = Enum('action', 'FORWARD BACKWARD TURN_LEFT TURN_RIGHT ABORT LOOK_UP LOOK_DOWN')

    action_loop_thread = None
    next_action = ''
    action_active = False

    # --- Servo things ---

    servo = Servo(15)
    MIN_DUTY = -1
    MAX_DUTY = 1
    MIN_ANGLE = 5
    MAX_ANGLE = 65
    servo_angle = 45

    # --- Stepper things ---
    # m1 is the left motor, m2 is the right one
    motors_enable = 14
    m1_step = 25
    m1_dir = 8
    m2_step = 23
    m2_dir = 24

    wheel_diameter = 76     # Diameter if wheel in mm
    steps_per_turn = 3200   # Steps needed to turn the wheel 360 degrees
    wheels_distance = 203   # Distance between wheels in mm

    def __init__(self):

        self.action_loop_thread = Thread(target=self.check_action)
        self.action_loop_thread.start()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.motors_enable, GPIO.OUT)
        GPIO.setup(self.m1_step, GPIO.OUT)
        GPIO.setup(self.m1_dir, GPIO.OUT)
        GPIO.setup(self.m2_step, GPIO.OUT)
        GPIO.setup(self.m2_dir, GPIO.OUT)

        GPIO.output(self.motors_enable, GPIO.LOW)
        #GPIO.output(self.m1_step, GPIO.LOW)
        #GPIO.output(self.m1_dir, GPIO.LOW)
        #GPIO.output(self.m2_step, GPIO.LOW)
        #GPIO.output(self.m2_dir, GPIO.LOW)

        self.servo.value = self.deg_to_duty(45)

    def deg_to_duty(self, deg):
        return (deg - self.MIN_ANGLE) * (self.MAX_DUTY - self.MIN_DUTY) / self.MAX_ANGLE + self.MIN_DUTY

    def servo_look_up(self, angle):
        self.servo_angle = min(self.servo_angle - angle, self.MAX_ANGLE)
        self.servo.value = min(self.deg_to_duty(self.servo_angle), 0)
        sleep(0.5)

    def servo_look_down(self, angle):
        self.servo_angle = max(self.servo_angle + angle, self.MIN_ANGLE)
        self.servo.value = max(self.deg_to_duty(self.servo_angle), 1)
        sleep(0.5)

    def stepper_control(self, m1_steps, m2_steps):

        # print("m1 steps: ")
        # print(m1_steps)
        # print("m2 steps: ")
        # print(m2_steps)

        if m1_steps > 0:
            GPIO.output(self.m1_dir, GPIO.HIGH)
        else:
            GPIO.output(self.m1_dir, GPIO.LOW)

        if m2_steps > 0:
            GPIO.output(self.m2_dir, GPIO.LOW)
        else:
            GPIO.output(self.m2_dir, GPIO.HIGH)

        m1_steps = abs(m1_steps)
        m2_steps = abs(m2_steps)

        for i in range(m1_steps):   # this assumes both motors will step an equal amount of times
            GPIO.output(self.m1_step, GPIO.HIGH)
            GPIO.output(self.m2_step, GPIO.HIGH)
            sleep(0.0001)
            GPIO.output(self.m2_step, GPIO.LOW)
            GPIO.output(self.m1_step, GPIO.LOW)
            sleep(0.0001)

    def stepper_control_dynamic(self, m1_steps, m2_steps):  # not really working as of right now :(

        print("m1 steps: ")
        print(m1_steps)
        print("m2 steps: ")
        print(m2_steps)

        if m1_steps > 0:
            GPIO.output(self.m1_dir, GPIO.HIGH)
        else:
            GPIO.output(self.m1_dir, GPIO.LOW)

        if m2_steps > 0:
            GPIO.output(self.m2_dir, GPIO.HIGH)
        else:
            GPIO.output(self.m2_dir, GPIO.LOW)

        m1_steps = abs(m1_steps)
        m2_steps = abs(m2_steps)

        for step_count in range(max(m1_steps, m2_steps)):
            # Step for at least the amount of steps from the stepper with most steps

            if step_count > m1_steps:
                GPIO.output(self.m1_step, GPIO.HIGH)
            if step_count > m2_steps:
                GPIO.output(self.m2_step, GPIO.HIGH)

            sleep(0.0001)

            GPIO.output(self.m1_step, GPIO.LOW)
            GPIO.output(self.m2_step, GPIO.LOW)

            sleep(0.0001)

    def angle_to_steps(self, angle):    # Steps needed to turn the robot a certain angle
        angle = abs(angle)     # make sure the angle is positive to make calculations correct
        turns = self.wheels_distance / self.wheel_diameter * (angle/360)
        return int(self.steps_per_turn * turns)

    def distance_to_steps(self, distance):
        rad = 2*distance/self.wheel_diameter
        return int(self.steps_per_turn * rad / (2*math.pi))

    def go_forward(self, distance):   # input distance in mm
        steps = self.distance_to_steps(distance)
        self.stepper_control(m1_steps=steps, m2_steps=steps)

    def go_backward(self, distance):
        steps = self.distance_to_steps(distance)
        self.stepper_control(m1_steps=-steps, m2_steps=-steps)

    def turn(self, angle):      # Angle in degrees
        steps = self.angle_to_steps(angle)
        # print("turn angle and steps:")
        # print(angle)
        # print(steps)
        if angle > 0:
            self.stepper_control(m1_steps=-steps, m2_steps=steps)
        else:
            self.stepper_control(m1_steps=steps, m2_steps=-steps)

    def check_action(self):  # Made to loop in a separate thread to monitor and perform actions
        print("Starting stepper control loop thread.")

        while True:

            if self.next_action == self.action.ABORT:
                print("Aborting stepper control loop thread.")
                break

            if self.next_action == self.action.FORWARD:
                self.go_forward(distance=100)
            elif self.next_action == self.action.BACKWARD:
                self.go_backward(distance=100)
            elif self.next_action == self.action.TURN_LEFT:
                self.turn(angle=45)
            elif self.next_action == self.action.TURN_RIGHT:
                self.turn(angle=-45)
            elif self.next_action == self.action.LOOK_UP:
                self.servo_look_up(10)
            elif self.next_action == self.action.LOOK_DOWN:
                self.servo_look_down(10)

            # robot.set_forward_distance(100)
            # if robot.is_ready
            #   robot.next_action = 'FORWARD'
            #   robot.go()

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


if __name__ == '__main__':
    robot = RobotControl()

    robot.stepper_control(-2000, -2000)

    sleep(2)

    robot.go_forward(100)

    robot.turn(90)

    pass
