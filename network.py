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

    def calculate_new_weight(self, error, pixel, learning_rate, old_weight):
        print("wtf")

    def get_label(self):
        return self.label
