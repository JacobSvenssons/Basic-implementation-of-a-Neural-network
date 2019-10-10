import numpy as np
from random import random
import sys


class ReadFile():
    def __init__(self, file):
        self.file = open(file, "r")  # insert  training-images.txt
        # Read the first two lines cause we do not need them.
        self.file.readline()
        self.file.readline()
        # read third row and fetch the given values

    def read_training(self):
        self.images, self.rows, self.cols, self.digits = map(int, self.file.readline().split())

        # Read the file and store every row (image) as an array inside an array
        img = []
        for i in range(images):
            line = list(map(int, file.readline().split()))
            line = np.array(list(map(0.001.__mul__, line)))
            img.append(line)
        img = np.array(img)

# TODO: From where will the "input" come from?
weights = [0.1 * random() - 0.05 for x in range(images)]
weights = np.array(weights)



for idx, pic in enumerate(img):
    print("Image:", idx + 1)
    print(pic)

