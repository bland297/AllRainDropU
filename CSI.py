# 예보, 관측
# A (Yes, Yes)
# B (No, Yes)
# C (Yes, No)
# D (No, No)

# HR(Hiting Rate) 적중률 : (A+D) / (A+B+C+D)

# POD 탐지율 : A / A+B 결과 YES에 대해 예측 YES의 비율 = (YES,YES) / (?,YES)
# FAR 부적중률 : C / A+C 예측 YES에 대해 결과 NO의 비율 = (YES,NO) / (YES,?)
# CSI 성공지수 : A / A+B+C 결과 YES+온다고했는데안옴 中 비가예보되고온날 = (YES,YES) / (?,YES)+(YES,NO)

from cmath import log10, nan
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import torch
import os
import random
import math

level = [0.1,1,5,10]
# level = np.log(np.array(level)+0.01)

data_dir = 'C:/Users/ks297/Desktop/AllRainDropUnetData/result/numpy'
lst_data = os.listdir(data_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_output = [f for f in lst_data if f.startswith('output')]
lst_label.sort()
lst_output.sort()


def compare(output, label):
    CSI=[]
    for i in range(len(level)):
        
        forecast = np.power(10,output)-0.01 >= level[i]
        observe = np.power(10,label)-0.01 >= level[i]
        hit_mask = forecast & observe
        total_mask = forecast | observe
        hit_num = np.sum(hit_mask)
        total_num = np.sum(total_mask)
        if total_num==0:
            csi = -1
        else :
            csi = hit_num/total_num
        CSI.append(np.round(csi,4))
    #print(CSI)
    return CSI

def print_csimean():
    csilist=[]
    for j in range(len(lst_label)):
        output = np.load(data_dir+'/'+lst_output[j])
        label = np.load(data_dir+'/'+lst_label[j])
        csilist += [compare(output,label)]
    csimean = [0,0,0,0]
    for j in range(4):
        sums = 0
        count = 0
        for k in range(len(csilist)):
            if csilist[k][j]!=-1:
                sums += csilist[k][j]
                count +=1
        if count == 0:
            csimean[j]=0
        else :
            csimean[j]=sums/count
    print(np.round(csimean,4))

def show_random():
    n=5
    fig, axes = plt.subplots(nrows=n, ncols=6)
    fig.suptitle('Random Display')
    cols = ['Predict','Real','CSI 0.1','CSI 1','CSI 5','CSI 10']
    for ax, col in zip(axes[0],cols):
        ax.set_title(col)

    for k in range(n):
        i = random.randint(0,len(lst_label))
        outputi = np.load(data_dir+'/'+lst_output[i])
        plt.subplot(n,6,1+6*k)
        plt.imshow(outputi, cmap='jet',vmin=-2,vmax=2)
        plt.axis('off')
        labeli = np.load(data_dir+'/'+lst_label[i])
        plt.subplot(n,6,2+6*k)
        plt.imshow(labeli, cmap='jet',vmin=-2,vmax=2)
        plt.axis('off')
        for j in range(len(level)):
            forecast = (np.power(10,outputi)-0.01 >= level[j])*2
            observe = (np.power(10,labeli)-0.01 >= level[j])*2
            hit_mask = (forecast & observe)/2
            total_mask = (forecast | observe)*1
            differ_makst = forecast - observe + hit_mask
            plt.subplot(n,6,j+6*k+3)
            plt.imshow(differ_makst, cmap='jet',vmin=-2,vmax=2)
            plt.axis('off')
            # unique,counts = np.unique(differ_makst, return_counts=True)
            # unique_dict = dict(zip(unique,counts))
            # print(unique_dict)
    plt.show()

def print_top5():
    csilist=[]
    for j in range(len(lst_label)):
        output = np.load(data_dir+'/'+lst_output[j])
        label = np.load(data_dir+'/'+lst_label[j])
        a = compare(output,label)
        for b in a:
            if b==1:
                print(j)
        csilist.append(a)
    arr = torch.tensor(csilist)
    arr = arr.transpose(1,0)
    arr = arr.tolist()
    top = []
    top_index = []
    for i in arr:
        best = []
        best_index = []
        for j in range(5):
            best.append(max(i))
            best_index.append(i.index(max(i)))
            i.remove(max(i))
        top.append(best)
        top_index.append(best_index)
    print(top,top_index)

    fig, axes = plt.subplots(nrows=4, ncols=5)
    fig.suptitle('Top5 Display')
    cols =  ['Top1','Top2','Top3','Top4','Top5']
    for ax, col in zip(axes[0],cols):
        ax.set_title(col)
    rows = ['CSI 0.1','CSI 1','CSI 5','CSI 10']
    for bx, row in zip(axes[:,0],rows):
        bx.set_ylabel(row)
    for i in range(4):
        for j in range(5):
            plt.subplot(4,5,j+1+5*i)
            plt.xlabel(top[i][j])
            plt.imshow(np.load(data_dir+'/'+lst_output[top_index[i][j]]),cmap='jet',vmin=-2.5,vmax=2.5)
            cx = plt.gca()
            cx.axes.xaxis.set_ticks([])
            cx.axes.yaxis.set_ticks([])
    plt.show()

def print_top5dif():
    csilist=[]
    for j in range(len(lst_label)):
        output = np.load(data_dir+'/'+lst_output[j])
        label = np.load(data_dir+'/'+lst_label[j])
        csilist += [compare(output,label)]
    arr = torch.tensor(csilist)
    arr = arr.transpose(1,0)
    arr = arr.tolist()
    top = []
    top_index = []
    for i in arr:
        best = []
        best_index = []
        for j in range(5):
            best.append(max(i))
            best_index.append(i.index(max(i)))
            i.remove(max(i))
        top.append(best)
        top_index.append(best_index)
    print(top,top_index)

    fig, axes = plt.subplots(nrows=4, ncols=5)
    fig.suptitle('Top5 Display')
    cols =  ['Top1','Top2','Top3','Top4','Top5']
    for ax, col in zip(axes[0],cols):
        ax.set_title(col)
    rows = ['CSI 0.1','CSI 1','CSI 5','CSI 10']
    for bx, row in zip(axes[:,0],rows):
        bx.set_ylabel(row)
    for i in range(4):
        for j in range(5):
            outputi = np.load(data_dir+'/'+lst_output[top_index[i][j]])
            labeli = np.load(data_dir+'/'+lst_label[top_index[i][j]])
            forecast = (np.power(10,outputi)-0.01 >= level[i])*2
            observe = (np.power(10,labeli)-0.01 >= level[i])*2
            hit_mask = (forecast & observe)/2
            differ_makst = forecast - observe + hit_mask
            plt.subplot(4,5,j+1+5*i)
            plt.xlabel(top[i][j])
            plt.imshow(differ_makst,cmap='jet',vmin=-2,vmax=2)
            cx = plt.gca()
            cx.axes.xaxis.set_ticks([])
            cx.axes.yaxis.set_ticks([])
    plt.show()
            
# show_random()
# print_csimean()
# print_top5()
print_top5dif()
