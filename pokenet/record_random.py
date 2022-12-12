from pyboy import PyBoy
from pyboy import WindowEvent
import random
from PIL import Image, ImageOps
from pokenet import ACTIONS

with PyBoy('Pmon', disable_renderer=False) as pyboy:
    pyboy.set_emulation_speed(0)
    pyboy.send_input(WindowEvent.STATE_LOAD)

    while True:
        #pyboy.screen_image().show()

        pyboy.tick()
        pyboy.tick()
        pyboy.tick()
        pyboy.tick()
        pyboy.tick()
        pyboy.tick()
        pyboy.tick()


        act = random.choice(ACTIONS)
        pyboy.send_input(act[0])
        pyboy.tick()
        pyboy.tick()
        pyboy.send_input(act[1])
        pyboy.tick()
        pyboy.tick()