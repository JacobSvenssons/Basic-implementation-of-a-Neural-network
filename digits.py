import numpy as np
from random import random
import sys


# Generate a list with random weights. Size is the size of every picture
# We want to create one for every network, so if we have 4,7,8 and 9 we need 4 lists with random weights.
# They will update after every "run" with a picture in a network.
def generate_weight_list(size):
    weights = [0.1 * random() - 0.05 for x in range(size)]
    weights = np.array(weights)
    return weights


# A FUNCTION FOR SPLITTING A LIST IN A DESIRED WAY
# RATIO IS A NUMBER 0-1 AND SETS THE "MID POINT" WHERE THE LIST SHOULD SPLIT
def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


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
    training_images, test_images = list_splitter(np_all_images, 0.75)
    training_labels, test_labels = list_splitter(np_all_labels, 0.75)
    shuffled_training_images, shuffled_training_labels = list_shuffler(training_images, training_labels)
    shuffled_test_images, shuffled_test_labels = list_shuffler(test_images, test_labels)

# Vi vill skapa en "perceptron" som ska "tränas" aka dennes weight lista uppdateras. En perceptron är ett objekt, men
    # kan också bara vara en weight lista som uppdateras "tränas"
    perceptron_4 = generate_weight_list(len(shuffled_training_images))
    perceptron_7 = generate_weight_list(len(shuffled_training_images))
    perceptron_8 = generate_weight_list(len(shuffled_training_images))
    perceptron_9 = generate_weight_list(len(shuffled_training_images))



    #   ------- TEST CASES -------
    ### A list if you want to test the function list_splitter ###

    #list1 = ['A', 'B', 'C', 'D', 'E', 'F']
    #a, b = list_splitter(list1, 0.75)
    #print(a)
    #print(b)


    ### TWO lists if you want to test the function list_shuffler ###
    # create a list, split the list and labels, use shuffle function and print them both
    # and see that they still match.

    #list1 = ['A', 'B', 'C', 'D', 'E', 'F']
    #list2 = [1, 2, 3, 4, 5, 6]
    #list1_training, list1_test = list_splitter(list1, 0.75)
    #list2_training, list2_test = list_splitter(list2, 0.75)
    #a, b = list_shuffler(list1_training, list2_training)
    #print(a)
    #print(b)


#for idx, pic in enumerate(img):
 #   print("Image:", idx + 1)
  #  print(pic)

