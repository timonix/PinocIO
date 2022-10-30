from time import sleep
import RPi.GPIO as GPIO
from sshkeyboard import listen_keyboard
import sshkeyboard


en = 14
step = 15
dir = 18

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
        for i in range(200):
            GPIO.output(step, GPIO.HIGH)
            sleep(0.005)
            GPIO.output(step, GPIO.LOW)
            sleep(0.005)

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
