import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from skimage import color
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class Parameters:
    train_set_path = os.path.join(os.getcwd(), 'GTSRB', 'Train')
    test_set_path = os.path.join(os.getcwd(), 'GTSRB', 'Test')
    test_set_annotation_filename = 'GT-final_test.csv'

    n_trees = 100

    resize_dim = (30, 30)
    track_length = 30

    brightness_prob = 0.5
    blur_prob = 0.5

    brightness_alpha_range = (0.6, 2.3)
    brightness_beta_range = (-15, 30)

    blur_kernel_range = (1, 3)
    # Number of images that will be blended together to generate a new image for augmentation
    augment_images_count = 10


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

        for c in range(43):
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

        self.training_data = train_data
        self.training_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def augment(self):
        classes = range(43)
        frequencies = [self.training_labels.count(c) for c in classes]
        target = max(frequencies)

        start_index = 0
        for i, c in enumerate(classes):
            end_index = start_index + frequencies[i]
            class_images = self.training_data[start_index:end_index]
            new_images = DataUtils.generate_images(class_images, target - len(class_images))

            self.training_labels += [c]*(target - len(class_images))
            self.training_data += new_images
            start_index = end_index

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
        for c in range(0, 43):
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
        shape = len(dataset), dataset[0].shape[0]*dataset[0].shape[1]
        res = np.zeros(shape, np.float)
        for i, img in enumerate(dataset):
            res[i] = DataUtils.normalize(img)

        return res

    @staticmethod
    def normalize(image):
        grayscale = color.rgb2gray(image)

        return grayscale.flatten()

    @staticmethod
    def generate_image(images, method='random'):
        """
        Generate an image out of given ones and change it randomly.
        Used in augmentation

        :param images: images to blend together to form a new image
        :param method: how to change the blended image. One of the options:
                      'brightness', 'blur', or 'random'
                      'brightness' mode transforms the image by alpha*x + beta,
                      where alpha and beta are chosen randomly from uniform distribution
                      'blur' applies blur kernel of random size to the picture
                      'random' applies a combination of previous modes (or none of them)
                      depending on the outcome of random variables

        :return: an image made by blending input images and applying random transformation to them
        """

        res_image = images[0]
        for i in range(1, len(images)):
            res_image = cv2.addWeighted(res_image, 0.5, images[i], 0.5, 0)

        # Methods: change_brightness (brightness), add_blur (blur)
        if method == 'random':
            probs = np.random.random(2)
            change_brightness = probs[0] < Parameters.brightness_prob
            add_blur = probs[1] < Parameters.blur_prob

            if change_brightness:
                res_image = DataUtils.generate_image([res_image], 'brightness')

            if add_blur:
                res_image = DataUtils.generate_image([res_image], 'blur')

            return res_image

        elif method == 'brightness':
            alpha = np.random.uniform(*Parameters.brightness_alpha_range)
            beta = np.random.uniform(*Parameters.brightness_beta_range)
            return cv2.convertScaleAbs(res_image, alpha=alpha, beta=beta)

        elif method == 'blur':
            kernel_size = np.random.randint(*Parameters.blur_kernel_range)
            return cv2.blur(res_image, (kernel_size, kernel_size))

        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

    @staticmethod
    def generate_images(images, image_count):
        res = []
        for i in range(image_count):
            indexes = np.random.choice(len(images), min(Parameters.augment_images_count, len(images)), replace=False)
            sample_images = [images[j] for j in indexes]
            np.random.sample()
            new_image = DataUtils.generate_image(sample_images)
            res.append(new_image)

        return res


class Model:
    def __init__(self):
        self.__classifiers = []

    def fit(self, train_data, train_labels):
        for i in range(Parameters.n_trees):
            classifier = DecisionTreeClassifier(max_features="sqrt")
            bs_x, bs_y = Model.__bootstrap(train_data, train_labels)
            classifier.fit(bs_x, bs_y)
            self.__classifiers.append(classifier)

    def predict(self, test_data):
        if not self.__classifiers:
            raise Exception("Model is not fit yet.")

        predictions = np.zeros((Parameters.n_trees, test_data.shape[0]), dtype="int")
        for i in range(Parameters.n_trees):
            predictions[i, :] = self.__classifiers[i].predict(test_data)

        prediction = mode(predictions, axis=0)[0].ravel()
        return prediction

    @staticmethod
    def accuracy_score(test_labels, predicted):
        return accuracy_score(test_labels, predicted)

    @staticmethod
    def __bootstrap(data, labels):
        index = np.random.randint(0, len(data), len(data))
        res_data = [data[i] for i in index]
        res_labels = [labels[i] for i in index]
        return np.array(res_data), np.array(res_labels)
