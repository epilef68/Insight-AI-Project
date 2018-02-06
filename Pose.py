#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:28:19 2018

@author: Felipe
"""


from config_reader import config_reader
import math
import cv2

import numpy as np
import util
from scipy.ndimage.filters import gaussian_filter

from keras.models import load_model

def heatmap2pose(heatmap,paf):
    param, model_params = config_reader()
#Identify peaks in heatmaps
    all_peaks = []
    peak_counter = 0
    
    part = 0
    for part in range(19-1):
        map_ori = heatmap[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
        
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    
    
    ############
    
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
              [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
              [55,56], [37,38], [45,46]]
    
    
    
    connection_all = []
    special_k = []
    mid_num = 10
    
    k=0
    
    for k in range(len(mapIdx)):
        score_mid = paf[:,:,[x-19 for x in mapIdx[k]]] # extract xy vectors of paf
        candA = all_peaks[limbSeq[k][0]-1]  # all 0 joints of limbSeq, like all wrists 
        candB = all_peaks[limbSeq[k][1]-1]  # all 1 joints of limbSeq, like all wrists 
        nA = len(candA) # number of possible wrists or elbows
        nB = len(candB) # number of possible wrists or elbows
        indexA, indexB = limbSeq[k] # index of particular joint
        if(nA != 0 and nB != 0): # skip if no enties for a joint
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2]) # calculate vector of possible limb
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)
                    
                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))) # create a list of points along limb
                    
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])
        
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
        
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break
        
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    
    
    
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1
    
            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print ("found = 2")
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
    
                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
                    
        
    full_list=[]
    dInd=    [1, 0,1,1,1,0,1,1,1,1,1,1, 1, 1, 1, 1, 1, 1]
    
    for n in range(len(subset)):
        count=0
        partList=tuple()
        for i in [12,1,0,2,3,4,4,5,6,7,8,9,10,11,13,15,14,16]:#in range(17):
        
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                partList=partList+(0,0)
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
    
            partList=partList+(X[dInd[count]],Y[dInd[count]])
            count=count+1
        full_list.append(partList)
    return full_list
     

def OpenPoseModel(oriImg,model):
    m=1
    # scale image
    param, model_params = config_reader()

    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
    scale = multiplier[m]
    
    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
   
    #pad image to have correct dimensions
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
   
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)     
      
    output_blobs = model.predict(input_img)
    
    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
    paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    full_list=heatmap2pose(heatmap,paf)
    
    return full_list

if __name__ == '__main__':
    
    test_image='sample_images/000000215259.jpg'

    oriImg = cv2.imread(test_image) # B,G,R order
    
    model = load_model('model/OpenPose.h5')
    full_list=OpenPoseModel(oriImg,model)
#    return full_list
    
    
    
    
    
    


