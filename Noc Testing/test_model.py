from train_model import NN
from train_model import input_size
from train_model import num_classes
import torch
from PIL import Image
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN(input_size=input_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model.m"))
model.eval()

test_image = Image.open("test_image.png")
test_dataset = np.asarray(test_image)

arr = np.asarray(test_image)
arr = np.moveaxis(arr, (2), (0))
arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
print(arr.shape)

output = model(torch.FloatTensor(arr))

with open("test_facit.json", "r") as facit:
    facit_dict = json.load(facit)

print("Output: ", output)
print("Facit: ", facit_dict["test_image.png"])
test_image.show()
