import os
import shutil

import pygame as pg
import random
from PIL import Image
import json
from tqdm import tqdm

pg.init()

#size = width, height = 300, 200
black = 0, 0, 0
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255
colors = [green, blue]

#min_radius = 5
#max_radius = 20



def generate_train_dataset(num_images, width, height):

    size = width, height

    directory = "../Train Images/"
    file_name_base = "train_image"
    file_type = ".png"

    #json_file_name = "Noc Testing/facit.json"

    facit_data = dict()

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in tqdm(range(num_images), desc="Generating train images"):
        screen, ai_position_x, ai_position_y, ai_radius = generate_image(size, width, height)
        file_name = directory + file_name_base + str(i) + file_type

        pg.image.save(screen, file_name)

                      #file_name)
        #facit_data[file_name] = [ai_position_x, ai_position_y, ai_radius]

        #with open(json_file_name, "w") as facit:
        #    json.dump(facit_data, facit)


def generate_test_dataset(num_images, width, height):

    size = width, height

    directory = "Test Images/"
    file_name_base = "test_image"
    file_type = ".png"

    facit_data = dict()

    for i in tqdm(range(num_images), desc="Generating test images"):

        screen, ai_position_x, ai_position_y, ai_radius = generate_image(size, width, height)
        file_name = directory + file_name_base + str(i) + file_type
        pg.image.save(screen, file_name)
        #facit_data[file_name] = [ai_position_x, ai_position_y, ai_radius]

        #with open("test_facit.json", "w") as facit:
        #    json.dump(facit_data, facit)


def generate_image(size, width, height):

    screen = pg.Surface(size)

    screen.fill(black)


    min_radius = int(width/15)
    max_radius = int(width/8)

    if min_radius <= 0:
        min_radius = 1

    if max_radius < 3:
        max_radius = 3

    for i in range(10):
        position = [random.randint(0, width), random.randint(0, height)]
        radius = random.randint(min_radius, max_radius)
        pg.draw.circle(screen, random.choice(colors), position, radius)

    position = [random.randint(0, width), random.randint(0, height)]
    radius = random.randint(min_radius, max_radius)

    # picture = Image.open(directory+file_name)
    # picture.save(directory+"Compressed_"+file_name, optimize=True, quality=30)
    ai_position_x = position[0] / width
    ai_position_y = position[1] / height
    ai_radius = (radius - min_radius) / (max_radius - min_radius)

    pg.draw.circle(screen, red, position, radius)

    return screen, ai_position_x, ai_position_y, ai_radius


if __name__ == "__main__":
    generate_train_dataset(10, 32, 32)
    #generate_test_dataset(10, 100, 50)
    pass

