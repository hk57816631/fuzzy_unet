#coding: utf-8
from keras.utils.np_utils import to_categorical
from PIL import Image
import numpy as np
import os

def label_to_categorical(path, classes):
    label_list = os.listdir(path)
    label_list.sort()
    for name in label_list:
        if name.endswith(".png"):
        # if name.endswith(".bmp"):
            img = Image.open(path + "/" + name)
            w = img.size[0]
            h = img.size[1]
            labels = np.array(img)
            labels[np.where(labels > classes - 1)] = classes - 1
            tmp = labels.flatten()
            #labels = np.ndarray((1,w*h, classes), dtype=np.uint8)
            categoricals = to_categorical(tmp, classes).reshape(h, w, classes)
            #print(name)
            #tmp = tmp.reshape(h, w)
            #img1 = Image.fromarray(tmp)
            #img1.putpalette(img.getpalette())
            #img1.show()

# save images' names in a txt file
def genfilelist(path, fname):
    train_list = os.listdir(path)
    train_list.sort()
    f1 = open(fname, "a")
    for name in train_list:
        if name.endswith(".png"):
        #if name.endswith(".jpg"):
            shotname, extension = os.path.splitext(name)
            print(path + "/" + shotname)
            #img = Image.open(path + "/" + name)
            #img.save(path + "/" + shotname + ".jpg")
            f1.writelines(shotname + "\n")
    f1.close()

# show the label image usage
class VOCPalette(object):
    def __init__(self, nb_class=21, start=1):
        self.palette = [0] * 768
        # voc2012 21palette
        if nb_class > 21 or nb_class < 2:
            nb_class = 21
        if start > 20 or start < 1:
            start = 1
        pal = self.labelcolormap(21)
        self.palette[0] = pal[0][0]
        self.palette[1] = pal[0][1]
        self.palette[2] = pal[0][2]
        for i in range(nb_class):
            self.palette[(i + 1) * 3] = pal[start][0]
            self.palette[(i + 1) * 3 + 1] = pal[start][1]
            self.palette[(i + 1) * 3 + 2] = pal[start][2]
            start = (start + 1) % 21
            if start == 0:
                start = 1
        assert len(self.palette) == 768

    def genlabelpal(self, img_arr):
        img = Image.fromarray(img_arr)
        img.putpalette(self.palette)

        return img

    def genlabelfilepal(self, path, isCoverLab):
        label_list = os.listdir(path)
        label_list.sort()
        for name in label_list:
            if name.endswith(".png"):
            #if name.endswith(".bmp"):
                img = Image.open(path + "/" + name).convert('L')
                shotname, extension = os.path.splitext(name)
                if isCoverLab == True:
                    img_arr = np.array(img)
                    img_arr[np.where(img_arr == 255)] = 1  # for 2 classes: 255->1
                    img = Image.fromarray(img_arr)
                img.putpalette(self.palette)
                img.save(path + "/" + shotname + ".png")

    def uint82bin(self, n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

    def labelcolormap(self, N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        for i in range(N):
            r = 0
            g = 0
            b = 0
            id = i
            for j in range(7):
                str_id = self.uint82bin(id)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap

if __name__ == '__main__':

    pal =VOCPalette(nb_class=5)
    pal.genlabelfilepal("../../BUS/VOC2012/GT_tumor",False)
