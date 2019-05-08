# This function is used to do wavelet transform
# three inputs: img_path: the path of original images; result_path: the path to save wavelet results; and txt_file:
# the txt file saving image names
from __future__ import print_function
from PIL import Image
import numpy as np
import cv2
import pywt
# the path of original images
img_path = './BUS/data1/original/'
# the path to save wavelet results
result_path = './BUS/data1/norm/'
# the txt file saving image names
txt_file = './BUS/data1/BUS.txt'

# open the txt file and read names
with open(txt_file,"r") as f:
    ls = f.readlines()
namesimg = [l.rstrip('\n') for l in ls]
nb_data_img = len(namesimg)

# do wavelet transform for every images in name list
for i in range(nb_data_img):
    # image whole path
    Xpath = img_path + "{}.png".format(namesimg[i])
    # print path
    print(Xpath)
    # # wavelet image generation
    # read images
    img = cv2.imread(Xpath, 0)
    # histogram equalization: map gray-level intensity to [0, 255]
    equ = cv2.equalizeHist(img, 256)
    # one level discret 2D wavelet transform on preprocessed images
    coeffs = pywt.dwt2(equ, 'haar')
    # wavelet coefficients
    # cA: approximate coefficient
    # (cH, cV, cD): details coefficients
    cA, (cH, cV, cD) = coeffs
    # map wavelet coefficients to the original size using bilinear interpolation
    C0 = cv2.resize(cA, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # map wavelet coefficients to the original size using bilinear interpolation
    C1 = cv2.resize(cH, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # map wavelet coefficients to the original size using bilinear interpolation
    C2 = cv2.resize(cV, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # map wavelet coefficients to the original size using bilinear interpolation
    C3 = cv2.resize(cD, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # combine three details coefficients using squared mean
    wavelet2 = np.square(C1) + np.square(C2) + np.square(C3)
    wavelet = np.sqrt(wavelet2)
    wavelet_new = np.zeros((img.shape[0],img.shape[1],3))

    # map the approximate and detail coefficients to [0, 255]
    C0min, C0max = C0.min(), C0.max()
    C01 = ((C0-C0min)/(C0max-C0min))*255
    waveletmin, waveletmax = wavelet.min(), wavelet.max()
    wavelet1 = ((wavelet-waveletmin)/(waveletmax-waveletmin))*255
    # construct a three channel image:
    # the first channel is the original image;
    # the second channel is the wavelet approximate coefficients;
    # the third channel is the wavelet details coefficients.
    wavelet_new[:, :, 0] = equ
    wavelet_new[:, :, 1] = C01
    wavelet_new[:, :, 2] = wavelet1
    # convert to uint8
    wavelet_new = np.uint8(wavelet_new)
    # save results
    imgorg = Image.fromarray(wavelet_new[:, :, 0])
    b = "{}.png".format(namesimg[i])
    imgorg.save(result_path + b)
