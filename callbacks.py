from __future__ import print_function
from keras.callbacks import Callback
from dataset_parser.prepareData import VOCPalette
from PIL import Image
import numpy as np
import os


class TrainCheck(Callback):
    def __init__(self, output_path, model_name, img_shape, nb_class):
        self.epoch = 0
        self.output_path = output_path
        self.model_name = model_name
        self.img_shape = img_shape
        self.nb_class = nb_class
        self.palette = VOCPalette(nb_class=nb_class)

    def result_map_to_img(self, res_map):
        res_map = np.squeeze(res_map)
        argmax_idx = np.argmax(res_map, axis=2).astype('uint8')

        return argmax_idx

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        # self.visualize(os.path.join(self.output_path,'test.png'))

    def visualize(self, path):
        imgorg = Image.open(path).convert('RGB')
        img = imgorg.resize((self.img_shape[1], self.img_shape[0]), Image.ANTIALIAS)
        img_arr = np.array(img)
        img_arr = img_arr / 127.5 - 1
        img_arr = np.expand_dims(img_arr, 0)
        pred = self.model.predict(img_arr)
        res_img = self.result_map_to_img(pred[0])

        PIL_img_pal = self.palette.genlabelpal(res_img)
        PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)
        PIL_img_pal.save(os.path.join(self.output_path, self.model_name + '_epoch_' + str(self.epoch) + '.png'))

