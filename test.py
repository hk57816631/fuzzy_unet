# this function is used to test one single images
#
from __future__ import print_function
import argparse
from PIL import Image
import numpy as np
from dataset_parser.prepareData import VOCPalette
from model.unet import unet
from model.fuzzy_unet import fuzzy_unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
import matplotlib.pyplot as plt


labelcaption = ['background', 'tumor', 'fat', 'mammary', 'muscle', 'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'Dining table', 'dog', 'horse', 'Motor bike', 'person', 'Potted plant',
                'sheep', 'sofa', 'train', 'monitor']


# compute the segmentation map by the probability output from the network
def result_map_to_img(res_map):
    res_map = np.squeeze(res_map)
    argmax_idx = np.argmax(res_map, axis=2).astype('uint8')
    return argmax_idx

# find what tissues existing in the image
def findObject(labelimg):
    counts = np.zeros(20,dtype=np.int32)
    str_obj = ''
    for i in range(20):
        counts[i] = np.sum(labelimg == i+1)
        # more than 500 pixels means the tissues exist
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

# testing image's path
img_path = './BUS/data2/wavelet/Case81.png'
# label path
label_path = './BUS/data2/GT/Case81.png'

# input size
img_width = 128
img_height = 128
nb_class = 2
channels = 3


# Create model
print('Creating network...\n')
if model_name == "fcn":
    model = fcn_8s(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                   lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "fuzzyunet":
    model = fuzzy_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-3, lr_decay=5e-4)

# load weights
try:
    model.load_weights(model_name + '_model_weight.h5')
except:
    print("You must train model and get weight before test.")

# Palette, used to show the result
palette = VOCPalette(nb_class=nb_class)
# print the testing image's name
print(img_path)
# read testing image
imgorg = Image.open(img_path)
# read label
imglab = Image.open(label_path)
# resize the input image to the input layer's size
img = imgorg.resize((img_width, img_height), Image.ANTIALIAS)
# convert to numpy array
img_arr = np.array(img)
# Centering helps normalization image (-1 ~ 1 value)
img_arr = img_arr / 127.5 - 1
# batch size is set to one
img_arr = np.expand_dims(img_arr, 0)
# img_arr = np.expand_dims(img_arr, 3)
# go forward of the network
pred = model.predict(img_arr)
# compute the maximum axis number in probability tensor, which represents the label assigned
# res is the final label map
res = result_map_to_img(pred[0])
# print color on label map use palette
PIL_img_pal = palette.genlabelpal(res)
# resize the result to the original size
PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)

obj = findObject(np.array(PIL_img_pal))

# plot figures
plt.ion()
plt.figure('Unet test')
plt.suptitle(img_path)
plt.subplot(1, 3, 1), plt.title('org')
plt.imshow(imgorg), plt.axis('off')
plt.subplot(1, 3, 2), plt.title(obj)
plt.imshow(PIL_img_pal), plt.axis('off')
plt.subplot(1, 3, 3), plt.title('label')
plt.imshow(imglab), plt.axis('off')

plt.show()
plt.close(1)



