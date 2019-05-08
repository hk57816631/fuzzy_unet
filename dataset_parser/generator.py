
import numpy as np
import random


from PIL import Image
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

def pre_processing(img):

    # Centering helps normalization image (-1 ~ 1 value)
    return img / 127.5 - 1


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        # set the image augment parameters
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args

# Use only multi classes.
def data_generator_dir(names, path_to_train, path_to_target, img_shape, nb_class, b_size, mode):

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = len(names)
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]
            name = names[idx]
            Xpath = path_to_train + "{}.png".format(name)
            ypath = path_to_target + "{}.png".format(name)
            x_img = load_data(Xpath, img_shape, mode="data")
            y_img = load_data(ypath, img_shape, mode="label")

            x.append(x_img)
            y.append(y_img)

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, binarylab(b_size, y_result, img_shape, nb_class)

                x.clear()
                y.clear()

def binarylab(b_size, y_img, img_shape, nb_class):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, img_shape[0], img_shape[1], nb_class))

    # For np.where calculation.
    for i in range(nb_class):
        mask = (y_img == i)
        result_map[:, :, :, i] = np.where(mask, 1, 0)

    return result_map
# load data
def load_data(path, img_shape, mode=None):
    img = Image.open(path)
    img = img.resize((img_shape[1],img_shape[0]))

    if mode=="original":
        return img

    if mode=="label":
        y = np.array(img, dtype=np.int32)
        mask = y == 255
        y[mask] = 0
        y = np.expand_dims(y, axis=-1)
        return y
    if mode=="data":
        X = image.img_to_array(img)

        return X
