import numpy as np
from random import random

file = open("givenData/training-images.txt", "r")
# Read the first two lines cause we do not need them.
file.readline()
file.readline()
# read third row and fetch the given values
images, rows, cols, digits = map(int, file.readline().split())

# Read the file and store every row (image) as an array inside an array
img = []
for i in range(images):
    line = list(map(int, file.readline().split()))
    img.append(line)
img = np.array(img)

# TODO: From where will the "input" come from?
weights = [0.1 * random() - 0.05 for x in range(int(input()))]
weights = np.array(weights)
