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


def compute_highest(num1, num2, num3, num4):

    if num1 > max(num2, num3, num4):
        return 4
    elif num2 > max(num3, num4):
        return 7
    elif num3 > num4:
        return 8
    else:
        return 9


class ReadFile:
    """
    Class implementation for reading ascii-based file format images.
    """

    def __init__(self, file):
        """
        Read the input .txt file and removes the first two lines of rubbish comments.
        :param file: The file to be read.
        """
        self.file = open(file, "r")
        self.file.readline()
        self.file.readline()

    def read_images(self):
        """
        Reads the third line from the image.txt file and extracts the values needed to understand
        the properties of the images that are stored later in the file. Then stores all images into
        a numpy array.
        :return: An array of images.
        """
        images, rows, cols, digits = map(int, self.file.readline().split())

        img = []
        for i in range(images):
            line = list(map(int, self.file.readline().split()))
            line = np.array(list(map(0.001.__mul__, line)))
            img.append(line)
        img = np.array(img)

        return img, rows, cols

    def read_label(self):
        """
        Reads the third line from the label.txt file and extracts the values needed to understand
        the properties of the labels that are stored later in the file. Then stores all labels into
        a numpy array.
        :return: An array of labels.
        """
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
    np_all_images, img_rows, img_cols = all_images.read_images()

    all_label = ReadFile(sys.argv[2])
    np_all_labels = all_label.read_label()

    validation_images = ReadFile(sys.argv[3])
    val_all_images, val_rows, val_cols = validation_images.read_images()

    training_images, test_images = list_splitter(np_all_images, 0.83)
    training_labels, test_labels = list_splitter(np_all_labels, 0.83)
    shuffled_training_images, shuffled_training_labels = list_shuffler(training_images, training_labels)
    shuffled_test_images, shuffled_test_labels = list_shuffler(test_images, test_labels)

    p_4 = Network(4, img_rows*img_cols)
    p_7 = Network(7, img_rows*img_cols)
    p_8 = Network(8, img_rows*img_cols)
    p_9 = Network(9, img_rows*img_cols)

    nets = [p_4, p_7, p_8, p_9]

    mean_error = 1
    while mean_error > 0.2:
        error = 0
        mean_error = 0

        for i, image in enumerate(shuffled_training_images):

            for net in nets:
                act = net.activation_function(net.dot_product(image))
                err = net.calculate_error(shuffled_training_labels[i], act)
                net.calculate_new_weight(err, image, 0.045)

        nets_ans = [0, 0, 0, 0]
        correct_ans = 0
        total_correct_ans = 0

        for k, img in enumerate(shuffled_test_images):

            for j, net in enumerate(nets):
                nets_ans[j] = net.activation_function(net.dot_product(img))
                error += np.abs(net.calculate_error(shuffled_test_labels[k], nets_ans[j]))

            correct_ans = compute_highest(nets_ans[0], nets_ans[1], nets_ans[2], nets_ans[3])

            if correct_ans == shuffled_test_labels[k]:
                total_correct_ans += 1

        right = (total_correct_ans / len(shuffled_test_labels)) * 100
        mean_error = error / (len(shuffled_test_images) * len(nets))
        print(right)
        print(mean_error)

    val_ans = [0, 0, 0, 0]
    val_correct_ans = 0

    for k, img in enumerate(val_all_images):

        for j, net in enumerate(nets):
            dot_p = net.dot_product(img)
            val_ans[j] = net.activation_function(dot_p)

        val_correct_ans = compute_highest(val_ans[0], val_ans[1], val_ans[2], val_ans[3])
        #print(val_correct_ans)
