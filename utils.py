import numpy as np
import cv2


def resize_image(image, size):
    image = cv2.resize(image, (size, size), cv2.INTER_NEAREST)
    return image


def resize_images(images, size):
    return np.array([cv2.resize(image, size, cv2.INTER_NEAREST) for image in images])


def concat_image(images):
    b, h, w, c = images.shape
    num_side = int(np.sqrt(b))
    image = np.vstack([np.hstack(images[i * num_side:(i + 1) * num_side]) for i in range(num_side)])
    return image


def save_image(file_name, image):
    image = np.array((image + 1) * 127.5, dtype="uint8")
    cv2.imwrite(file_name, image)


def save_images(dir_name, images):
    images = np.array((images + 1) * 127.5, dtype="uint8")
    for i, image in enumerate(images):
        cv2.imwrite("{}/{}.jpg".format(dir_name, i), image)


