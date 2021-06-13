import gzip
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


def read_from_mnist(index):
    mndata = MNIST('samples')

    #images, labels = mndata.load_training()
    images, labels = mndata.load_testing()

    # index = random.randrange(0, len(images))  # choose an index ;-)
    print(mndata.display(images[index]))


def show_image_from_gzip():
    f = gzip.open('samples/train-images-idx3-ubyte.gz', 'r')
    image_size = 28
    num_images = 5
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    image = np.asarray(data[2]).squeeze()
    plt.imshow(image)
    plt.show()


def read_label_from_gzip(idx):
    f = gzip.open('samples/train-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    for i in range(idx, idx):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        print(labels)


def read_labels_from_gzip_in_range(idx_start, idx_end):
    f = gzip.open('samples/train-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    for i in range(idx_start, idx_end):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        print(labels)


read_from_mnist(0)
#read_label_from_gzip(5)
#read_labels_from_gzip_in_range(0, 5)
#show_image_from_gzip()
