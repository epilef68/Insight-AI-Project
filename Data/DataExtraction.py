#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:51:47 2018

@author: Felipe
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import scipy.io as sio
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/home/paperspace/Downloads/annotations/'
dataType='train2017'
annFile='/home/paperspace/Downloads/annotations/instances_train2017.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

#filter relevant data
catIds = coco.getCatIds(catNms=['person','couch']);
imgIds = coco.getImgIds(catIds=catIds );
imgSize=368
stride=8
numImag=min(len(imgIds),100)

I_full= np.zeros([numImag,imgSize, imgSize,3])

labelSize=21#46 #
paf_full=np.zeros([numImag, labelSize, labelSize,38])
heatmap_full=np.zeros([numImag, labelSize, labelSize,19])
count=0
count1=1
# initialize COCO api for person keypoints annotations
annFile = '/home/paperspace/Downloads/annotations/person_keypoints_train2017.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)


for imgId in imgIds:
    print(count/numImag)
    # load images
    img = coco.loadImgs(imgId)[0]
    I = io.imread(img['coco_url'])
    # reshape black and white images
    if len(I.shape)<3:
        BW=np.ones([I.shape[0],I.shape[1],3])
        BW[:,:,0]= I
        BW[:,:,1]= I
        BW[:,:,2]= I
        I=BW

    annIds = coco_kps.getAnnIds(imgId, catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)

    x=[]
    y=[]
    for ann in anns:
        kp = np.array(ann['keypoints'])
        x.append(kp[0::3])
        y.append(kp[1::3])


    bodyInd= (0, 14, 15, 16, 17, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
    full_list=[(0,0,0,0) for i in range(len(x[0])+1)]

    heatmaps = np.zeros((I.shape[0],I.shape[1],len(x[0])+2))
    pafs = np.zeros((I.shape[0],I.shape[1],2*(len(x[0])+2)))
    xCord = np.linspace(1, I.shape[1],I.shape[1])
    yCord = np.linspace(1, I.shape[0],I.shape[0])

    xx, yy = np.meshgrid(xCord,yCord)
    sig = 30

    for joint in range(len(x[0])):
        partList=tuple()
        heatmap = np.zeros((I.shape[0],I.shape[1]))


        for person in range(len(x)):

            partList = partList+(x[person][joint],y[person][joint])

            rCord = np.sqrt((x[person][joint] - xx)**2 + (y[person][joint] - yy)**2)
            gauss = np.exp((-rCord**2) / (2 * sig**2))
            if x[person][joint]==0 & y[person][joint]==0:
                gauss=np.zeros(gauss.shape)
            heatmap=np.maximum(heatmap ,gauss)

        full_list[bodyInd[joint]]=partList

        heatmaps[:,:,bodyInd[joint]]=heatmap

     # find connection in the specified sequence

    limbSeq = [[2,9], [9,10], [10,11],[2,12], [12,13], [13,14], \
               [2,3], [3,4], [4,5], [3,17], [2,6], [6,7], [7,8],\
               [6,18],[2,1], [1,15], [15,17], [1,16], [16,18]]


    limbWidth=10 # limb length of mask in pixels

    pafs = np.zeros((I.shape[0],I.shape[1],2*(len(x[0])+2)))

    for limb in range(0,len(pafs[1][1]),2):

        if sum(full_list[limbSeq[int(limb/2)][0]-1])==0:
            continue
        if sum(full_list[limbSeq[int(limb/2)][1]-1])==0:
            continue

        X1=full_list[limbSeq[int(limb/2)][0]-1][0::2] #limb1
        X2=full_list[limbSeq[int(limb/2)][1]-1][0::2] #limb2
        Y1=full_list[limbSeq[int(limb/2)][0]-1][1::2] #limb1
        Y2=full_list[limbSeq[int(limb/2)][1]-1][1::2] #limb2
        for person in range(len(x)):

            VX=(X2[person]-X1[person])
            VY=(Y2[person]-Y1[person])
            VV=np.sqrt(VX**2+VY**2)
            VX=VX/VV
            VY=VY/VV
            xOffset=xx-X1[person]
            yOffset=yy-Y1[person]
            dot = xOffset*VX+yOffset*VY
            orth=(xOffset**2+yOffset**2)-dot**2
            orth[orth<0]=0
            orthDist=np.sqrt(orth)
            pafx = np.ones((I.shape[0],I.shape[1], 1))
            pafy = np.ones((I.shape[0],I.shape[1], 1))
            pafx=pafx*VX
            pafy=pafy*VY
            pafx[orthDist>limbWidth]=0
            pafy[orthDist>limbWidth]=0

            xOffset2=xx-X2[person]
            yOffset2=yy-Y2[person]
            r1=np.sqrt(xOffset**2+yOffset**2)
            r2=np.sqrt(xOffset2**2+yOffset2**2)

            pafx[np.logical_or(r1>VV,r2>VV)]=0
            pafy[np.logical_or(r1>VV,r2>VV)]=0
            pafs[:,:,limb]=pafs[:,:,limb]+np.squeeze(pafx)
            pafs[:,:,limb+1]=pafs[:,:,limb+1]+np.squeeze(pafy)
            # add averaging of overlaping limbs


    I_full[count][:][:][:]=cv2.resize(I,(imgSize,imgSize))

    paf_full[count][:][:][:]=cv2.resize(pafs,(labelSize,labelSize))
    heatmap_full[count][:][:][:]=cv2.resize(heatmaps,(labelSize,labelSize))

    count=count+1
    if count>=numImag:
       count=0
       mdict ={'I_full':I_full, 'paf_full':paf_full,'heatmap_full':heatmap_full}
       sio.savemat('labeledDataTrain3_{}.mat'.format(count1), mdict)
       count1=count1+1





# save results to MAT files
mdict = {'I_full':I_full,'paf_full':paf_full,'heatmap_full': heatmap_full}

sio.savemat('labeledDataTrain.mat', mdict)
