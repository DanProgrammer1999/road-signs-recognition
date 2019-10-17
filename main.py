import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Parameters:
    train_set_path = os.path.join(os.getcwd(), "GTSRB", "Train")
    test_set_path = os.path.join(os.getcwd(), "GTSRB", "Test")
    test_set_annotation_filename = "GT-final_test.csv"

    resize_dim = (30, 30)
    track_length = 30


class DataUtils:
    def __init__(self):
        self.training_data = None
        self.training_labels = None
        self.testing_data = None
        self.testing_labels = None

    def load_data(self, training=True, testing=True):
        if training:
            self.training_data, self.training_labels = self.get_train_data()
        if testing:
            self.testing_data, self.testing_labels = self.get_test_data()

        return self

    def transform_data(self):
        if self.training_data:
            self.training_data = list(map(DataUtils.transform_image, self.training_data))
        if self.testing_data:
            self.testing_data = list(map(DataUtils.transform_image, self.testing_data))

    def train_validation_split(self, train_portion=0.8):
        tracks_indexes = np.array(range(len(self.training_data) // Parameters.track_length))

        np.random.shuffle(tracks_indexes)

        train_size = int(train_portion * len(tracks_indexes))

        train_data, train_labels, validation_data, validation_labels = [], [], [], []

        for i in range(len(tracks_indexes)):
            curr_data = self.training_data[i * Parameters.track_length:
                                           i * Parameters.track_length + Parameters.track_length]
            curr_labels = self.training_labels[i * Parameters.track_length:
                                               i * Parameters.track_length + Parameters.track_length]
            if i < train_size:
                train_data += curr_data
                train_labels += curr_labels
            else:
                validation_data += curr_data
                validation_labels += curr_labels

        return train_data, train_labels, validation_data, validation_labels

    @staticmethod
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
                    labels.append(int(row[7]))  # the 8th column is the label

        return images, labels

    @staticmethod
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
                images.append(
                    plt.imread(os.path.join(Parameters.test_set_path, row[0])))  # the 1th column is the filename
                labels.append(int(row[7]))  # the 8th column is the label

        return images, labels

    @staticmethod
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
            paddings += [0] * (len(arr.shape) - 2)

        paddings = tuple((padding // 2, padding - (padding // 2)) for padding in paddings)

        res = np.pad(arr, paddings, 'constant')

        return res

    @staticmethod
    def resize(arr):
        res = Image.fromarray(arr)
        res = res.resize(Parameters.resize_dim)
        return np.array(res)

    @staticmethod
    def transform_image(arr):
        return DataUtils.resize(DataUtils.make_square(arr))
