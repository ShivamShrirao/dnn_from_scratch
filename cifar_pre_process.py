import numpy as np
import pickle
from gc import collect
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def one_hot_encode(vec, vals=10):
    n = len(vec)
    y = np.zeros((n, vals))
    y[range(n), vec] = 1
    return y


def normalize(x):
    mn = x.min()
    mx = x.max()
    x = (x - mn) / (mx - mn)
    return x


# Class to handel the dataset
class CifarPreProcess():
    def __init__(self, CIFAR_DIR='../cifar-10-batches-py/'):
        self.CIFAR_DIR = CIFAR_DIR
        files = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
                 'test_batch']

        self.all_data = []

        for i, fname in enumerate(files):
            self.all_data.append(unpickle(self.CIFAR_DIR + fname))

        self.names = self.all_data[0][b'label_names']
        for i, nm in enumerate(self.names):
            self.names[i] = nm.decode("utf-8")
        self.st = 0
        self.train_len = 0

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

        self.itr_train = None

    def set_up_images(self):

        print("Setting Up Training Images and Labels")

        # Vertically stacks the training images
        self.training_images = np.vstack([d[b'data'] for d in self.all_data[1:-1]])
        self.train_len = self.training_images.shape[0]
        self.training_images = normalize(self.training_images.reshape(self.train_len, 3, 32, 32).transpose(0, 2, 3, 1))
        self.training_images = self.training_images.astype(np.float32)
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_data[1:-1]]), 10)

        print("Setting Up Test Images and Labels")

        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in [self.all_data[-1]]])
        test_len = self.test_images.shape[0]
        self.test_images = normalize(self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1))
        self.test_images = self.test_images.astype(np.float32)
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in [self.all_data[-1]]]), 10)
        self.shuffle_datasets()
        self.all_data = None
        collect()

    def shuffle_datasets(self):
        idxs = np.arange(len(self.training_images))
        np.random.shuffle(idxs)
        self.training_images = self.training_images[idxs]
        self.training_labels = self.training_labels[idxs]
        idxs = np.arange(len(self.test_images))
        np.random.shuffle(idxs)
        self.test_images = self.test_images[idxs]
        self.test_labels = self.test_labels[idxs]

    def data_augment(self, batch_size=64):
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        self.datagen.fit(self.training_images)
        self.itr_train = self.datagen.flow(self.training_images, self.training_labels, batch_size=batch_size)
        collect()
        return self.itr_train

    def make_dataset_from_iterator(self):
        if self.itr_train is None:
            return None
        steps = self.training_images.shape[0] // self.batch_size
        INP = np.empty((steps, self.batch_size, *self.training_images.shape[1:]), dtype=np.float32)
        LBL = np.empty((steps, self.batch_size, *self.training_labels.shape[1:]), dtype=np.float32)
        for idx in range(steps):
            try:
                INP[idx], LBL[idx] = self.itr_train.next()
            except ValueError:
                INP[idx], LBL[idx] = self.itr_train.next()
        return INP.reshape(-1, *self.training_images.shape[1:]), LBL.reshape(-1, *self.training_labels.shape[1:])

    def batch_gen(self, size, ck=0):
        if not ck:
            ck = self.st
            self.st = (self.st + size) % self.train_len
        x = self.training_images[ck:ck + size].reshape(-1, 32, 32, 3)
        y = self.training_labels[ck:ck + size]
        return x, y
