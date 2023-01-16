import sshkeyboard
from sshkeyboard import listen_keyboard
from threading import Thread
import robot_control
import RPi.GPIO as GPIO
from time import sleep

robot = robot_control.RobotControl()


def press(key):
    #print(f"'{key}' pressed")

    if key == "w":
        robot.next_action = robot.action.FORWARD
    elif key == "s":
        robot.next_action = robot.action.BACKWARD
    elif key == "a":
        robot.next_action = robot.action.TURN_LEFT
    elif key == "d":
        robot.next_action = robot.action.TURN_RIGHT
    elif key == 'i':
        robot.next_action = robot.action.LOOK_UP
    elif key == 'k':
        robot.next_action = robot.action.LOOK_DOWN
    elif key == 'q':
        robot.next_action = robot.action.ABORT
        sshkeyboard.stop_listening()  # Shutdown is here to make check_action have a chance to shut itself down


def release(key):
    robot.next_action = ''


if __name__ == "__main__":

    print("Start of manual control \n Use WASD to control the robot, press q to abort")

    #t_check_input = Thread(target=robot.check_action)
    #t_check_input.start()

    # Listen for inputs
    listen_keyboard(
        on_press=press,
        on_release=release,
        # until='q',              # Abort when q is pressed
    )

    GPIO.cleanup()

    print("Program aborted. Shutting down. Good day.")




