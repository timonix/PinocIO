from time import sleep
import RPi.GPIO as GPIO

en = 14
step = 15
dir = 18


GPIO.setmode(GPIO.BCM)
GPIO.setup(en, GPIO.OUT)
GPIO.setup(step, GPIO.OUT)
GPIO.setup(dir, GPIO.OUT)

GPIO.output(en, LOW)
GPIO.output(step, LOW)
GPIO.output(dir, LOW)

try:
    while True:
        GPIO.output(step, HIGH)
        sleep(0.1)
        GPIO.output(step, LOW)
        sleep(0.1)

except KeyboardInterrupt:
    GPIO.output(en, LOW)
    GPIO.output(step, LOW)
    GPIO.output(dir, LOW)
    GPIO.cleanup()
    pass
