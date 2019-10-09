import numpy as np

file = open("givenData/training-images.txt", "r")
# Read the first two lines cause we do not need them.
file.readline()
file.readline()
#läs in tredje raden för att hämta värderna
images, rows, cols, digits = map(int, file.readline().split())

img = []

# Läser filen och lagrar varje rad in en array som lagras i en annan array
for i in range(images):
    line = list(map(int, file.readline().split()))
    img.append(line)
print(img[2][150])

#weights = np.array()

