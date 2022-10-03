import os
import random
import uuid

import numpy as np
from vidgear.gears import CamGear
from cunker import cunker
import cv2
from PIL import Image

from os import listdir
from os.path import isfile, join


# generate test cunks from youtube videos



class YTExtractor():

    def __init__(self, url='https://www.youtube.com/watch?v=E2sSvVCRI4s', framePercentage=1.0, path="../Train Images/"):
        stream = CamGear(source=url, stream_mode=True, logging=True).start()  # YouTube Video URL as input
        frame = stream.read()

        print("--------------")
        print(stream.ytv_metadata["title"])

        num_frames = stream.ytv_metadata["duration"] * stream.ytv_metadata["fps"]

        print("STARTING")
        loop_num = 0
        while frame is not None:
            loop_num += 1

            if loop_num % 1000 == 0:
                print(loop_num, end="/")
                print(num_frames)

            if random.random() < framePercentage:
                color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_coverted)
                img = pil_image
                cunk_size = 64

                cc = cunker(cunk_size)
                for c in cc.cunk(img):
                    c.save(path + str(uuid.uuid4()) + ".png")

            frame = stream.read()

        print("LOADED")


class Data():
    def __init__(self, path="../Train Images/"):
        self.path = path
        self.files = [f for f in listdir(path) if f.endswith(".png") and isfile(join(path, f))]
        self.index = 0

    def getNext(self, batch_size):
        start = self.index
        stop = min(self.index + batch_size, len(self.files))

        if start == stop:
            return None

        self.index = stop
        batch_files = self.files[start:stop]
        return [np.array(Image.open(self.path+fname)) for fname in batch_files]

    def shuffle(self):
        random.shuffle(self.files)

#
#ex = YTExtractor(url="https://www.youtube.com/watch?v=rc2urzXP4ok", framePercentage=1,path="C:\\Users\\u057742.CORP\\Train Images\\")