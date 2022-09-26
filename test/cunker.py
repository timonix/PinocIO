import cv2
import numpy as np
from PIL import Image


def __ressample__(arr, N):
    if len(arr[0][0]) % N != 0 or len(arr[0]) % N != 0:
        print("cropping image")
        print(str(len(arr[0][0])) + "x" + str(len(arr[0])) + " is not divisible by " + str(N) + "x" + str(N))

    num_horizontal_cunks = int(len(arr[0][0]) / N)
    num_vertical_cunks = int(len(arr[0]) / N)

    ret = [[None for _ in range(num_horizontal_cunks)] for _ in range(num_vertical_cunks)]

    for h in range(num_horizontal_cunks):
        for v in range(num_vertical_cunks):
            ret[h][v] = arr[:, N * v:N * (v + 1), N * h:N * (h + 1)]

    return ret


class cunker():
    def __init__(self, cunk_size=32):
        self.cunk_size = cunk_size

    def cunk(self, im):
        im_width, im_height = im.size
        row_width = self.cunk_size
        row_height = self.cunk_size

        cols = int(im_height / self.cunk_size)
        rows = int(im_width / self.cunk_size)
        n = 0
        cunked_images = []
        for i in range(0, cols):
            for j in range(0, rows):
                box = (j * row_width, i * row_height, j * row_width +
                       row_width, i * row_height + row_height)
                outp = im.crop(box)
                cunked_images.append(outp)
        return cunked_images


class decunker():
    def __init__(self, image_size, cunk_size=32):
        self.cunk_size = cunk_size
        self.image_size = image_size

    def decunk(self, cunked_image):

        images_to_merge = cunked_image
        image1 = images_to_merge[0]
        cols = int(self.image_size[0]/self.cunk_size)
        rows = int(self.image_size[1]/self.cunk_size)
        new_width = self.cunk_size * cols
        new_height = self.cunk_size *rows
        new_image = Image.new(image1.mode, (new_width, new_height))
        for i in range(0, rows):
            for j in range(0, cols):
                image = images_to_merge[i * cols + j]
                new_image.paste(image, (j * image.size[0], i * image.size[1]))

        return new_image


def cunk_test():
    img = Image.open("C:\\Users\\tjade\\PycharmProjects\\PinocIO\\Train Images\\Cunk_test.bmp")

    cam = cv2.VideoCapture(0)
    ret_val, cv_img = cam.read()

    print(cv_img.shape)

    color_coverted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    img = pil_image

    image_size = (640, 480)
    cunk_size = 32

    cc = cunker(cunk_size)
    dc = decunker(image_size)

    ccc = cc.cunk(img)
    print(len(ccc))
    dc.decunk(ccc).show()


#cunk_test()
