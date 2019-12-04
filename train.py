from __future__ import print_function

import os

import argparse
from keras.callbacks import ModelCheckpoint
from callbacks import TrainCheck

from model.unet import unet
from model.fuzzy_unet import fuzzy_unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.SCFnet import SCFnet
from dataset_parser.generator import data_generator_dir

# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet', 'fuzzyunet', 'SCFnet'],
                    help="Model to train. 'fcn', 'unet', 'pspnet', 'fuzzyunet', 'SCFnet'is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=4, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=1, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=1e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")

args = parser.parse_args()
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg

# batch size
TRAIN_BATCH = 8
VAL_BATCH = 1

# epoch
epochs = 80
# continue training
resume_training = False

# path of document
# load the txt files contain names of images in train set
path_to_train = './BUS/data2/BUS1.txt'
path_to_val = './BUS/data2/BUS2.txt'
path_to_img = './BUS/data2/wavelet/'
path_to_label = './BUS/data2/GT/'

# the size of input layer
# image size
img_width = 128
img_height = 128

# category number
nb_class = 2

# input image channel
channels = 3

# read the name in train set
with open(path_to_train,"r") as f:
    ls = f.readlines()
namestrain = [l.rstrip('\n') for l in ls]
nb_data_train = len(namestrain)
# read the name in validation set
with open(path_to_val,"r") as f:
    ls = f.readlines()
namesval = [l.rstrip('\n') for l in ls]
nb_data_val = len(namesval)

# Create model to train
print('Creating network...\n')
if model_name == "fcn":
    model = fcn_8s(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "fuzzyunet":
    model = fuzzy_unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "SCFnet":
    model = SCFnet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)


# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_name + '_model_checkpoint_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)
train_check = TrainCheck(output_path='./img', model_name=model_name, img_shape=(img_height,img_width),nb_class=nb_class)

# load weights if needed
if resume_training:
    print('Resume training...\n')
    model.load_weights(model_name + '_model_checkpoint_weight.h5')
else:
    print('New training...\n')

# training
history = model.fit_generator(data_generator_dir(namestrain, path_to_img, path_to_label,(img_height, img_width, channels), nb_class, TRAIN_BATCH, 'train'),
                              steps_per_epoch=nb_data_train // TRAIN_BATCH,
                              validation_data=data_generator_dir(namesval, path_to_img, path_to_label, (img_height, img_width, channels), nb_class, VAL_BATCH, 'val'),
                              validation_steps=nb_data_val // VAL_BATCH,
                              callbacks=[checkpoint,train_check],
                              epochs=epochs,
                              verbose=1)

# serialize model weigths to h5
model.save_weights(model_name + '_model_weight.h5')
# save the training loss and validation loss to txt
f_loss = open("loss.txt","a")
f_loss.write(str(history.history))
f_loss.close()
