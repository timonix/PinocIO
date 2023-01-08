from time import sleep
import RPi.GPIO as GPIO
from sshkeyboard import listen_keyboard
import sshkeyboard


#en = 25
step1 = 23
dir1 = 24

step2 = 25
dir2 = 8

print("Starting setup")

GPIO.setmode(GPIO.BCM)
#GPIO.setup(en1, GPIO.OUT)
GPIO.setup(step1, GPIO.OUT)
GPIO.setup(dir1, GPIO.OUT)

GPIO.setup(step2, GPIO.OUT)
GPIO.setup(dir2, GPIO.OUT)

#GPIO.output(en, GPIO.LOW)
GPIO.output(step1, GPIO.LOW)
GPIO.output(dir1, GPIO.LOW)

GPIO.output(step2, GPIO.LOW)
GPIO.output(dir2, GPIO.LOW)


def press(key):
    if key == 's':
        print("stepping")
        for i in range(1600):
            GPIO.output(step1, GPIO.HIGH)
            GPIO.output(step2, GPIO.HIGH)
            sleep(0.01)
            GPIO.output(step1, GPIO.LOW)
            GPIO.output(step2, GPIO.LOW)
            sleep(0.01)

#    if key == 'l':
#        GPIO.output(en, GPIO.LOW)
#    if key == 'h':
#        GPIO.output(en, GPIO.HIGH)

    if key == 'q':
        sshkeyboard.stop_listening()  # Shutdown is here to make check_action have a chance to shut itself down


print("Starting stepping")

listen_keyboard(
        on_press=press,
        # on_release=release,
        # until='q',              # Abort when q is pressed
)

print("Interrupted")
GPIO.output(en, GPIO.LOW)
GPIO.output(step, GPIO.LOW)
GPIO.output(dir, GPIO.LOW)
GPIO.cleanup()

#try:
#    while True:
#        GPIO.output(step, GPIO.HIGH)
#        sleep(0.1)
#        GPIO.output(step, GPIO.LOW)
#        sleep(0.1)

#except KeyboardInterrupt:
#    print("Interrupted")
#    GPIO.output(en, GPIO.LOW)
#    GPIO.output(step, GPIO.LOW)
#    GPIO.output(dir, GPIO.LOW)
#    GPIO.cleanup()
#    pass
