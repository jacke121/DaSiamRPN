# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python
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
vi=cv2.VideoCapture(0)

cx, cy, w, h = 400,300,100,300

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
ret, image = vi.read()
print(image.shape)
state = SiamRPN_init(image, target_pos, target_sz, net)  # init tracker
while True:
    ret, image = vi.read()
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
    cv2.waitKey(1)

    # handle.report(Rectangle(res[0], res[1], res[2], res[3]))

