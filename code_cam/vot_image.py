


import  os

import time

import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

path=r"D:\data\vot2017\ants1"
# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track

for file in os.listdir(path):
    if file.endswith(".jpg"):
        image = cv2.imread(path+"/"+file)
        break
#找一张图片直接调一下即可
cx, cy, w, h = 150.21,500,50,90

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

print(image.shape)
state = SiamRPN_init(image, target_pos, target_sz, net)  # init tracker
for file in os.listdir(path):
    if file.endswith(".jpg"):
        image = cv2.imread(path+"/"+file)
    # frame = imread(sequence_path+"%04d.jpg"%i)
        i += 1
        if image is None:
            break
        time1=time.time()
        state = SiamRPN_track(state, image)  # track
        print("{time:.3f}s".format(time=time.time() - time1))
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1=int(res[0])
        y1=int(res[1])
        width=int(res[2])
        height=int(res[3])
        imshow = cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
        cv2.imshow("imshow", imshow)
        cv2.waitKeyEx()

    # handle.report(Rectangle(res[0], res[1], res[2], res[3]))

