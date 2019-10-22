import csv
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.utils import shuffle
import seaborn as sn
import pandas as pd


class Parameters:
    train_set_path = os.path.join(os.getcwd(), 'GTSRB', 'Train')
    test_set_path = os.path.join(os.getcwd(), 'GTSRB', 'Test')
    test_set_annotation_filename = 'GT-final_test.csv'

    n_trees = 512
    max_depth = 500
    min_samples_leaf = 5
    criterion = "gini"

    resize_dim = (30, 30)
    enable_augment = True

    # Number of images that will be blended together to generate a new image for augmentation
    augment_images_count = 10

    brightness_prob = 0.5

    brightness_alpha_range = (0.5, 2)
    brightness_beta_range = (-10, 10)

    # Modify only if you change the dataset
    track_length = 30
    class_count = 43


class DataUtils:
    def __init__(self):
        self.training_data = None
        self.training_labels = None
        self.testing_data = None
        self.testing_labels = None
        self.validation_data = None
        self.validation_labels = None

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
        train_data, train_labels = [], []
        validation_data, validation_labels = [], []
        start_index = 0

        for c in range(Parameters.class_count):
            class_count = self.training_labels.count(c)
            end_index = start_index + class_count

            track_indexes = list(range(start_index, end_index, Parameters.track_length))
            np.random.shuffle(track_indexes)

            train_size = int(np.ceil(len(track_indexes) * train_portion))

            for i, index in enumerate(track_indexes):
                if i < train_size:
                    train_data += self.training_data[index: index + Parameters.track_length]
                else:
                    validation_data += self.training_data[index: index + Parameters.track_length]

            train_labels += [c] * train_size * Parameters.track_length
            validation_labels += [c] * (len(track_indexes) - train_size) * Parameters.track_length
            start_index = end_index

        train_data, train_labels = shuffle(train_data, train_labels)
        validation_data, validation_labels = shuffle(validation_data, validation_labels)

        self.training_data = train_data
        self.training_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def augment(self):
        classes = range(Parameters.class_count)
        frequencies = [self.training_labels.count(c) for c in classes]
        target = max(frequencies)
        images = [[] for _ in range(Parameters.class_count)]

        for i, label in enumerate(self.training_labels):
            images[label].append(self.training_data[i])

        for i, im_class in enumerate(images):
            generate_count = target - len(im_class)
            self.training_data += DataUtils.generate_images(im_class, generate_count)
            self.training_labels += [i] * generate_count

        return True

    def normalize_images(self):
        self.training_data = DataUtils.normalize_set(self.training_data)
        if self.validation_data:
            self.validation_data = DataUtils.normalize_set(self.validation_data)
        self.testing_data = DataUtils.normalize_set(self.testing_data)

    @staticmethod
    def get_train_data():
        """Reads traffic sign training data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign training data, for example './GTSRB/Train'
        Returns:   list of images, list of corresponding labels
        """

        images = []  # images
        labels = []  # corresponding labels
        # loop over all 43 classes
        for c in range(0, Parameters.class_count):
            prefix = os.path.join(Parameters.train_set_path, format(c, '05d'))  # subdirectory for class
            csv_filename = 'GT-' + format(c, '05d') + '.csv'

            with open(os.path.join(prefix, csv_filename)) as csv_file:  # annotations file
                csv_reader = csv.reader(csv_file, delimiter=';')  # csv parser for annotations file
                next(csv_reader)  # skip header
                # loop over all images in current annotations file
                for row in csv_reader:
                    # images.append('Class {}, track {}, image {}'.format(c, int(row[0][:5]), int(row[0][6:-4])))
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
        return cv2.resize(arr, Parameters.resize_dim)

    @staticmethod
    def transform_image(arr):
        return DataUtils.resize(DataUtils.make_square(arr))

    @staticmethod
    def normalize_set(dataset):
        # Assuming shape is the same for all images
        shape = len(dataset), dataset[0].shape[0] * dataset[0].shape[1] * dataset[0].shape[2]
        res = np.zeros(shape, np.float)
        for i, img in enumerate(dataset):
            res[i] = DataUtils.normalize(img)

        return res

    @staticmethod
    def normalize(image):
        # res = color.rgb2gray(image)
        res = image / np.max(image)
        return res.flatten()

    @staticmethod
    def generate_image(images):
        """
        Generate an image out of given ones and change it randomly.
        Used in augmentation

        :param images: images to blend together to form a new image

        :return: an image made by blending input images and applying random transformations to them
        """

        res_image = images[0]
        for i in range(1, len(images)):
            res_image = cv2.addWeighted(res_image, 0.5, images[i], 0.5, 0)

        # Methods: change_brightness (brightness), add_blur (blur)
        prob = np.random.random(2)
        change_brightness = prob[0] < Parameters.brightness_prob

        if change_brightness:
            alpha = np.random.uniform(*Parameters.brightness_alpha_range)
            beta = np.random.uniform(*Parameters.brightness_beta_range)
            return cv2.convertScaleAbs(res_image, alpha=alpha, beta=beta)

        return res_image

    @staticmethod
    def generate_images(images, image_count):
        res = []
        for i in range(image_count):
            indexes = np.random.choice(len(images), min(Parameters.augment_images_count, len(images)))
            sample_images = [images[j] for j in indexes]
            np.random.sample()
            new_image = DataUtils.generate_image(sample_images)
            res.append(new_image)

        return res


class Model:
    def __init__(self):
        self.__classifier = None

    def fit(self, train_data, train_labels):
        self.__classifier = RandomForestClassifier(
            max_features="auto",
            n_estimators=Parameters.n_trees,
            n_jobs=-1,
            criterion=Parameters.criterion,
            min_samples_leaf=Parameters.min_samples_leaf,
            max_depth=Parameters.max_depth
        )
        self.__classifier.fit(train_data, train_labels)
        train_pred = self.__classifier.predict(train_data)
        return accuracy_score(train_labels, train_pred)

    def predict(self, data):
        return self.__classifier.predict(data)


class Test:

    @staticmethod
    def build_freq_chart(training_labels):
        labels = list(range(Parameters.class_count))
        freqs = [training_labels.count(label) for label in labels]
        plot = plt.bar(labels, freqs)
        plt.ylabel("Frequencies")
        plt.xlabel("Class")
        plt.title("Number of examples per each class")

        return plot

    @staticmethod
    def timed_function(f, *args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        return res, end - start

    @staticmethod
    def load_data(data: DataUtils):
        data, curr_time = Test.timed_function(data.load_data)
        print("Load time {:0.2f}s".format(curr_time))

    @staticmethod
    def preprocess_data(data: DataUtils):
        _, curr_time = Test.timed_function(data.transform_data)
        print("Transform: {:0.2f}s".format(curr_time))

        _, curr_time = Test.timed_function(data.train_validation_split)
        print("Split: {:0.2f}s".format(curr_time))

        if Parameters.enable_augment:
            _, curr_time = Test.timed_function(data.augment)
            print("Augment: {:0.2f}s".format(curr_time))
        else:
            print("Augment disabled by config")

        _, curr_time = Test.timed_function(data.normalize_images)
        print("Normalization: {:0.2f}s".format(curr_time))

    @staticmethod
    def train_model(data, model):
        score, curr_time = Test.timed_function(model.fit, data.training_data, data.training_labels)
        print("Train set accuracy: {}. \nModel fit: {:0.2f}s".format(score, curr_time))

        return model

    @staticmethod
    def get_predicted(dataset, model):
        predicted, curr_time = Test.timed_function(model.predict, dataset)
        print("Prediction: {:0.2f}s".format(curr_time))
        return predicted

    @staticmethod
    def print_scores(real, predicted, label: str):
        accuracy = accuracy_score(real, predicted)
        precision = precision_score(real, predicted, average='macro')
        recall = recall_score(real, predicted, average='macro')
        print("{} set accuracy: {:0.2f}%, precision: {:0.2f}%, recall: {:0.2f}%".
              format(label, accuracy * 100, precision * 100, recall * 100))

    @staticmethod
    def show_confusion_matrix(real, predicted, label: str):
        matrix = confusion_matrix(real, predicted, labels=list(range(Parameters.class_count)))
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        df = pd.DataFrame(matrix)
        plot = sn.heatmap(df, xticklabels=3, yticklabels=3)  # font size
        y_lim = list(plot.get_ylim())
        y_lim[0] += 0.5
        y_lim[1] = 0
        plot.set_ylim(y_lim)
        plt.title("Confusion matrix for {} set".format(label))
        plt.show()

        return matrix

    @staticmethod
    def main():
        data = DataUtils()
        model = Model()

        start = time.time()

        Test.load_data(data)
        Test.preprocess_data(data)
        Test.train_model(data, model)

        val_predicted = Test.get_predicted(data.validation_data, model)
        test_predicted = Test.get_predicted(data.testing_data, model)
        print("validation:", val_predicted)
        print("test:", test_predicted)
        Test.print_scores(data.validation_labels, val_predicted, 'validation')
        Test.print_scores(data.testing_labels, test_predicted, 'test')

        end = time.time()
        print("Total time: {:0.2f}s".format(end - start))


if __name__ == '__main__':
    Test.main()
