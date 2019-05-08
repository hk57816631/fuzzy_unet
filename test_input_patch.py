# this file is the same as test.py only different it tests a set of samples, I will not write detail comments
from __future__ import print_function
import argparse
from PIL import Image
import numpy as np
from dataset_parser.prepareData import VOCPalette

from model.unet import unet
from model.fuzzy_unet import fuzzy_unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
import cv2
import scipy.io as sio


labelcaption = ['background','tumor','fat','mammary','muscle','bottle','bus','car','cat',
                'chair','cow','Dining table','dog','horse','Motor bike','person','Potted plant',
                'sheep','sofa','train','monitor']
def result_map_to_img(res_map):
    res_map = np.squeeze(res_map)
    argmax_idx = np.argmax(res_map, axis=2).astype('uint8')

    return argmax_idx

def findObject(labelimg):
    counts = np.zeros(20,dtype=np.int32)
    str_obj = ''
    for i in range(20):
        counts[i] = np.sum(labelimg == i+1)
        if counts[i] > 500 :
            str_obj = str_obj + labelcaption[i+1]+ ' '
    return str_obj

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet', 'fuzzyunet'],
                    help="Model to train. 'fcn', 'unet', 'pspnet', 'fuzzyunet'is available.")
parser.add_argument("-P", "--img_path", required=False, help="The image path you want to test")

args = parser.parse_args()
model_name = args.model
img_path = args.img_path
vgg_path = None

# path of images
img_path = './BUS/data2/wavelet/'
# path of ground truth
label_path = './BUS/data2/GT/'
# path of the txt file contain name list
test_file = './BUS/data2/BUS.txt'
# result path one for image
result_path = './result/image/'
# save the probability matrix for further usage
result_path2 = './result/mat/'
img_width = 256
img_height = 256
nb_class = 2
channels = 3

# Create model to train
print('Creating network...\n')
if model_name == "fcn":
    model = fcn_8s(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                   lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "fuzzyunet":
    model = fuzzy_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
try:
    model.load_weights(model_name + '_model_weight_9.h5')
except:
    print("You must train model and get weight before test.")

palette = VOCPalette(nb_class=nb_class)

with open(test_file,"r") as f:
    ls = f.readlines()
namesimg = [l.rstrip('\n') for l in ls]
nb_data_img = len(namesimg)

for i in range(nb_data_img):
    Xpath = img_path + "{}.png".format(namesimg[i])
    Ypath = label_path + "{}.png".format(namesimg[i])

    print(Xpath)
    # read image
    imgorg = Image.open(Xpath)
    imglab = Image.open(Ypath)
    img = imgorg.resize((img_width, img_height), Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = img_arr / 127.5 - 1
    img_arr = np.expand_dims(img_arr, 0)
    img_arr = img_arr.reshape((1, img_arr.shape[1], img_arr.shape[2], 1))
    # feed the network
    pred = model.predict(img_arr)
    pred_test = pred[0]
    pred_test_res = cv2.resize(pred_test, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
    res = result_map_to_img(pred[0])
    # save the probability as .mat
    dataNew = result_path2+"{}.mat".format(namesimg[i])
    sio.savemat(dataNew, {'A': pred_test_res})
    PIL_img_pal = palette.genlabelpal(res)
    PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)
    b = "{}.png".format(namesimg[i])
    PIL_img_pal.save(result_path + b)




