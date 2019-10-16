import numpy as np
from random import random
import sys


class Network:
    """
    A class representing each neuron in the neural network.
    """

    def __init__(self, label, size):
        """
        Construct a neuron with a number it should specialize on.
        Also create an array of random weights for between the neuron and following node.
        :param label: The number the specialize on.
        :param size: The number of pixels in an image.
        """
        self.label = label
        self.weights = np.array([0.1 * random() - 0.05 for _ in range(size)])

    def calculate_error(self, label, a):
        """
        Calculate the error between the desired output y and the activation a of the current neuron.
        :param label: The label representing the real number of the current image.
        :param a: The activation function output.
        :return: The error value.
        """
        y = 1 if self.label == label else -1
        return y - a

    def calculate_new_weight(self, error, img, learning_rate):
        """
        Calculates the new weight between the neuron and the output node.
        :param error: The error between the desired output y and the activation function
        :param img: An image array with all its pixels
        :param learning_rate: Balance how much the weight is updated
        :return: Updated weight between the neuron and the following node.
        """
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + img[i] * error * learning_rate

    def activation_function(self, dot):
        """
        Maps the neurons inputs to corresponding output.
        :param dot: The dot product of the neurons weights and the pixel values.
        :return: A Value representing how active the neuron should be for this particular input.
        """
        return np.tanh(dot)

    def dot_product(self, pixels):
        """
        Calculate the dot product between all pixels and weights for an image.
        :param pixels: Array of pixels in an image
        :return: The dot product sum
        """
        return np.dot(self.weights, pixels)



def list_splitter(list_to_split, ratio):
    """
    Split the list of all images into test and training sub sets.
    :param list_to_split: Numpy array of all the images of handwritten numbers
    :param ratio: The proportion between the test and training sub set. Sets the midpoint for the split.
    :return: A list of testing images and a list of training images
    """
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


# A FUNCTION FOR SHUFFLING THE OBJECTS IN TWO LISTS THE SAME WAY.
# AKA CORRECT LABEL FOR THE IMAGE
def list_shuffler(image_list_to_shuffle, label_to_shuffle):
    """
    Takes in two lists of equal size and shuffles each list randomly. However both lists
    are shuffled in the same random order.
    :param image_list_to_shuffle: List of images
    :param label_to_shuffle: List of labels for the images.
    :return:
    """
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
