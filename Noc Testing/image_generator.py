import pygame as pg
import random
from PIL import Image
import json

pg.init()

size = width, height = 300, 200
black = 0, 0, 0
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255
colors = [green, blue]

min_radius = 5
max_radius = 20

screen = pg.Surface(size)


iterations = 20000
directory = "Images/"
file_name_base = "image"
file_type = ".png"

json_file_name = "facit.json"

dica = dict()


def generate_train_dataset(num_images):
    for i in range(0, num_images):
        screen.fill(black)

        for j in range(0, 10):
            position = [random.randint(0, width), random.randint(0, height)]
            radius = random.randint(min_radius, max_radius)
            pg.draw.circle(screen, random.choice(colors), position, radius)

        position = [random.randint(0, width), random.randint(0, height)]
        radius = random.randint(5, 20)

        pg.draw.circle(screen, red, position, radius)

        file_name = directory+file_name_base+str(i)+file_type

        pg.image.save(screen, file_name)

        # picture = Image.open(directory+file_name)
        # picture.save(directory+"Compressed_"+file_name, optimize=True, quality=30)
        ai_position_x = position[0] / width
        ai_position_y = position[1] / height
        ai_radius = (radius-min_radius) / (max_radius-min_radius)

        dica[file_name] = [ai_position_x, ai_position_y, ai_radius]

    with open(json_file_name, "w") as facit:
        json.dump(dica, facit)


def generate_test_dataset(num_images):

    for x in range(num_images):

        screen.fill(black)

        for j in range(0, 10):
            position = [random.randint(0, width), random.randint(0, height)]
            radius = random.randint(min_radius, max_radius)
            pg.draw.circle(screen, random.choice(colors), position, radius)

        position = [random.randint(0, width), random.randint(0, height)]
        radius = random.randint(5, 20)

        pg.draw.circle(screen, red, position, radius)

        file_name = "test_image.png"

        pg.image.save(screen, file_name)

        # picture = Image.open(directory+file_name)
        # picture.save(directory+"Compressed_"+file_name, optimize=True, quality=30)
        ai_position_x = position[0] / width
        ai_position_y = position[1] / height
        ai_radius = (radius - min_radius) / (max_radius - min_radius)

        dica[file_name] = [ai_position_x, ai_position_y, ai_radius]

        with open("test_facit.json", "w") as facit:
            json.dump(dica, facit)

