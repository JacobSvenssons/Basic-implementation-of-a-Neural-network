import numpy as np
from random import random
import sys


class Network:
    def __init__(self, label, size):
        self.label = label
        self.weights = np.array([0.1 * random() - 0.05 for _ in range(size)])

    def calculate_error(self, label, a):
        y = 1 if self.label == label else -1
        return y - a

    def calculate_new_weight(self, error, img, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + img[i] * error * learning_rate

    def activation_function(self, dot):
        return np.tanh(dot)

    def dot_product(self, pixels):
        return np.dot(self.weights, pixels)


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

    p_4 = Network(4, len(training_images[0]))
    p_7 = Network(7, len(training_images[0]))
    p_8 = Network(8, len(training_images[0]))
    p_9 = Network(9, len(training_images[0]))

    nets = [p_4, p_7, p_8, p_9]

    i = 0
    for image in shuffled_training_images:

        for net in nets:
            dot_p = net.dot_product(image)
            act = net.activation_function(dot_p)
            err = net.calculate_error(shuffled_training_labels[i], act)
            net.calculate_new_weight(err, image, 0.01)
        i += 1

    nets_ans = [0, 0, 0, 0]
    correct_ans = 0
    total_correct_ans = 0
    k = 0
    for img in shuffled_test_images:
        j = 0
        for net in nets:
            dot_p = net.dot_product(img)
            nets_ans[j] = net.activation_function(dot_p)
            j += 1

        if nets_ans[0] > max(nets_ans[1], nets_ans[2], nets_ans[3]):
            correct_ans = 4
        elif nets_ans[1] > max(nets_ans[2], nets_ans[3]):
            correct_ans = 7
        elif nets_ans[2] > nets_ans[3]:
            correct_ans = 8
        else:
            correct_ans = 9

        if correct_ans == shuffled_test_labels[k]:
            total_correct_ans += 1

        print(correct_ans, "VS", shuffled_test_labels[k])
        k += 1

    print(total_correct_ans/250 * 100)
