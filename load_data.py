import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.datasets import load_digits
import pandas as pd


def read_mnist(caracteres=[]):
    if not (all(x in range(0, 10) for x in caracteres)):
        print("MNIST digits range from 0 to 9.\n")
        return None
    
    mnist = load_digits()
    df = pd.DataFrame(mnist.data)
    indices = [index for index, target in enumerate(mnist.target) if target in caracteres]
    df = df.iloc[indices]
    df[:] = np.where(df < 8, 0, 1)

    return np.array(df), 8, 8


def show_mnist(digits):
    image, heigth, width = read_mnist([digits])

    for i in range(5):
        x = image[i].reshape((heigth, width))
        show(x)


def read_alpha_digit(path, chars=[]):
    full_data_json = scipy.io.loadmat(path)
    data_4d = full_data_json['dat'][chars]
    data_4d = data_4d.reshape(np.prod(data_4d.shape))
    height, width = data_4d[0].shape
    data_3d = [data_4d.reshape(np.prod(data_4d.shape)) for data_4d in data_4d]

    return np.array(data_3d), height, width


def show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def save_rbm(rbm, specs):
    file_name = 'models/trained_rbm'

    for key, value in specs.items():
        file_name += '_' + key + '_' + str(value)
    pickle.dump(rbm, open(file_name, "wb"))


def load_rbm(specs):
    file_name = 'models/trained_rbm'

    for key, value in specs.items():
        file_name += '_' + key + '_' + str(value)
    return pickle.load(open(file_name, "rb"))