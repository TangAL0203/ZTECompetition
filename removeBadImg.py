# -*- coding:utf-8 -*-
import os
import os.path as osp
from PIL import Image 

trainRoot = './test'
for typename in os.listdir(trainRoot):
    for imgname in os.listdir(osp.join(trainRoot,typename)):
        imgpath = osp.join(trainRoot,typename,imgname)
        try:
            Image.open(imgpath)
        except:
            os.remove(imgpath)

