import numpy as np
from random import random
import sys


class ReadFile:
    def __init__(self, file):
        self.file = open(file, "r")  # insert  training-images.txt
        # Read the first two lines cause we do not need them.
        self.file.readline()
        self.file.readline()
        # read third row and fetch the given values

    def read_images(self):
        images, rows, cols, digits = map(int, self.file.readline().split())

        # Read the file and store every row (image) as an array inside an array
        img = []
        for i in range(images):
            line = list(map(int, self.file.readline().split()))
            line = np.array(list(map(0.001.__mul__, line)))
            img.append(line)
        img = np.array(img)

        return img

    def read_label(self):
        images, digits = map(int, self.file.readline().split())

        # Read the file and store every row (image) as an array inside an array
        label = []
        for i in range(images):
            line = self.file.readline().strip()
            label.append(int(line))
        label = np.array(label)

        return label

# Vill vi eventuellt lagra labels och images i samma struktur med en kry/value?
if __name__ == "__main__":

    all_images = ReadFile(sys.argv[1])
    temp = all_images.read_images()

    all_label = ReadFile(sys.argv[2])
    temp2 = all_label.read_label()

    # Dela listorna i två delar, training och test, rekommenderat training = 75 / test = 25 som start.
    # Måste hålla koll så att labels och images är i samma ordning. Tar vi plats 10 i en lista
    # måste vi ävem göra det i andra så de har samma plats. Värderna måste shufflas runt,
    # borde bli samma sak med att ta random objekt och flytta till nya listorna.

    # För training, skapa ny array som initaliseras med random weights mellan - 0.05 till 0.05. Blir lika stor
    # som trainigs storlek. Sedan vill vi skicka in en pixel och dess weight i en
    # "activation function" samt en label compare ( se om den är rätt eller inte)
    # för att träna och uppdatera med en ny weight.


# TODO: From where will the "input" come from?
#weights = [0.1 * random() - 0.05 for x in range(images)]
#weights = np.array(weights)



#for idx, pic in enumerate(img):
 #   print("Image:", idx + 1)
  #  print(pic)

