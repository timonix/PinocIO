from time import sleep
import RPi.GPIO as GPIO
from sshkeyboard import listen_keyboard
import sshkeyboard


en = 25
step = 23
dir = 24

print("Starting setup")

GPIO.setmode(GPIO.BCM)
GPIO.setup(en, GPIO.OUT)
GPIO.setup(step, GPIO.OUT)
GPIO.setup(dir, GPIO.OUT)

GPIO.output(en, GPIO.LOW)
GPIO.output(step, GPIO.LOW)
GPIO.output(dir, GPIO.LOW)


def press(key):
    if key == 's':
        print("stepping")
        for i in range(1600):
            GPIO.output(step, GPIO.HIGH)
            sleep(0.01)
            GPIO.output(step, GPIO.LOW)
            sleep(0.01)

    if key == 'l':
        GPIO.output(en, GPIO.LOW)
    if key == 'h':
        GPIO.output(en, GPIO.HIGH)

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