import numpy as np
import matplotlib.pyplot as plt
import csv
import os


class Parameters:
    train_set_path = os.path.join(os.getcwd(), "GTSRB", "Train")
    test_set_path = os.path.join(os.getcwd(), "GTSRB", "Test")
    test_set_annotation_filename = "GT-final_test.csv"


def get_train_data():
    """Reads traffic sign training data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign training data, for example './GTSRB/Train'
    Returns:   list of images, list of corresponding labels
    """

    images = []  # images
    labels = []  # corresponding labels
    # loop over all 43 classes
    for c in range(0, 43):
        prefix = os.path.join(Parameters.train_set_path, format(c, '05d'))  # subdirectory for class
        csv_filename = 'GT-' + format(c, '05d') + '.csv'

        with open(os.path.join(prefix, csv_filename)) as csv_file:  # annotations file
            csv_reader = csv.reader(csv_file, delimiter=';')  # csv parser for annotations file
            next(csv_reader)  # skip header
            # loop over all images in current annotations file
            for row in csv_reader:
                images.append(plt.imread(os.path.join(prefix, row[0])))  # the 1th column is the filename
                labels.append(row[7])  # the 8th column is the label

    return images, labels


def get_test_data():
    """
    Reads traffic sign test data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign test data, for example './GTSRB/Test'
    Returns:   list of images, list of corresponding labels
    """

    images = []
    labels = []

    with open(os.path.join(Parameters.test_set_path, Parameters.test_set_annotation_filename)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')  # csv parser for annotations file
        next(csv_reader)  # skip header
        # loop over all images in current annotations file
        for row in csv_reader:
            images.append(plt.imread(os.path.join(Parameters.test_set_path, row[0])))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label

    return images, labels


def make_square(arr):
    """
    Pad first 2 dimensions of array to a square shape with padding equally spread to both sides
    :param arr: the array to be padded
    :return:    square-shaped array padded with 0's
    """
    if arr.shape[0] == arr.shape[1]:
        return arr

    longest_side = max(arr.shape[0:2])

    paddings = [(longest_side - side) for side in arr.shape[0:2]]
    if len(arr.shape) > 2:
        paddings += [0]*(len(arr.shape) - 2)

    paddings = tuple((padding//2, padding - (padding//2)) for padding in paddings)

    res = np.pad(arr, paddings, 'constant')

    return res
