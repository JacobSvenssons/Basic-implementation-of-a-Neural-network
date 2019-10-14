import numpy as np
from random import random
import sys


# A FUNCTION FOR SPLITTING A LIST, CURRENTLY ONLY IN HALF
# WILL UPDATE SO YOU CAN SPLIT IN A DESIRED WAY

# FRÅGA HANDLEDARE OM MAN MÅSTE SPLITTA LISTAN MER "RANDOM"? MEN EFTERSOM VI HAR 4 OLIKA SIFFROR SÅ KÄNNS DET
# SPONTANT SOM ATT DET RÄCKER ATT GÖRA SÅ HÄR??
def list_splitter(list_to_split):
    half = len(list_to_split) // 2
    return list_to_split[:half], list_to_split[half:]


# A FUNCTION FOR SHUFFLING THE OBJECTS IN TWO LISTS THE SAME WAY.
# AKA CORRECT LABEL FOR THE IMAGE
def list_shuffler(image_list_to_shuffle, label_to_shuffle):
    temp_zip = list(zip(image_list_to_shuffle, label_to_shuffle))
    np.random.shuffle(temp_zip)
    shuffled_image_list, shuffled_label_list = zip(*temp_zip)
    return shuffled_image_list, shuffled_label_list


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


if __name__ == "__main__":

    all_images = ReadFile(sys.argv[1])
    np_all_images = all_images.read_images()

    all_label = ReadFile(sys.argv[2])
    np_all_labels = all_label.read_label()

    # take the two lists and split them in training and test images,
    # and lastly shuffle the lists.
    training_images, test_images = list_splitter(np_all_images)
    training_labels, test_labels = list_splitter(np_all_labels)
    shuffled_training_images, shuffled_training_labels = list_shuffler(training_images, training_labels)
    shuffled_test_images, shuffled_test_labels = list_shuffler(test_images, test_labels)

    ### A list if you want to test the function list_splitter ###

    #list1 = ['A', 'B', 'C', 'D', 'E', 'F']
    #a, b = list_splitter(list1)
    #print(a)
    #print(b)

    ### TWO lists if you want to test the function list_shuffler ###

    #list1 = ['A', 'B', 'C', 'D', 'E', 'F']
    #list2 = [1, 2, 3, 4, 5, 6]
    #a, b = list_shuffler(list1, list2)
    #print(a)
    #print(b)

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

