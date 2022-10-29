import sshkeyboard
from sshkeyboard import listen_keyboard
from threading import Thread
import motor_control
from time import sleep

robot = motor_control.Movement()


def press(key):
    print(f"'{key}' pressed")
    robot.action = key

    if key == 'q':
        sshkeyboard.stop_listening()        # Shutdown is here to make check_action have a chance to shut itself down

#    if key == 'w':
#        movement.do_movement(movement='forward')
#    elif key == 'a':
#        movement.do_movement(movement='left')
#    elif key == 's':
#        movement.do_movement(movement='backward')
#    elif key == 'd':
#        movement.do_movement(movement='right')


def release(key):
    robot.action = ''


if __name__ == "__main__":

    print("Start of manual control \n Use WASD to control the robot, press q to abort")

    t_check_input = Thread(target=robot.check_input)
    t_check_input.start()

    # Listen for inputs
    listen_keyboard(
        on_press=press,
        on_release=release,
        # until='q',              # Abort when q is pressed
    )

    print("Program aborted. Shutting down. Good day.")




