import torch

MODEL_PATH = "C:\\Users\\tjade\\Documents\\pokedata\\models/"
DATA_PATH = "C:\\Users\\tjade\\Documents\\pokedata\\data/"

if torch.cuda.is_available():
    MODEL_PATH_RUNPOD = "models/"
    DATA_PATH_RUNPOD = "data/"