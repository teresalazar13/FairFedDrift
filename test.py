from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)

    return arr


if __name__ == '__main__':
    (train_X_priv, train_y_priv), (__, _) = mnist.load_data()
    image = train_X_priv[0]
    plt.imshow(image, cmap='gray')
    plt.show()
    print(image)

    new_image = train_X_priv[0]
    plt.imshow(new_image * -1, cmap='gray')
    plt.show()
    print(new_image)

    new_image = color_grayscale_arr(image, red=True)
    plt.imshow(new_image, cmap='gray')
    plt.show()
    print(new_image)

    new_image = color_grayscale_arr(image, red=False)
    plt.imshow(new_image, cmap='gray')
    plt.show()
    print(new_image)
