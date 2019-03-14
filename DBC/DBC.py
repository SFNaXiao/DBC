# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:46:09 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from sklearn import metrics
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
import hungalg

filename=input("Enter the file's name: ")
location=[]
##label=[]
Receivers=[]
for line in open(filename,"r"):
    items=line.strip("\n").split(" ")
    ##label.append(int(items.pop()))
    tmp=[]
    for item in items:
        tmp.append(float(item))
    location.append(tmp)
    Receivers.append([])
location=np.array(location)
##label=np.array(label)
length=len(location)

K=input("Please input the number of neighbors:")
K=int(K)

nbrs=NearestNeighbors(n_neighbors=K).fit(location)
distances,Neighbors=nbrs.kneighbors(location)

'''
dist=np.zeros((length,length))
begin=0
while begin<length-1:
    end=begin+1
    while end<length:
        dd=np.linalg.norm(location[begin]-location[end])
        dist[begin][end]=dd
        dist[end][begin]=dd
        end=end+1
    begin=begin+1
'''

vector={}
for i in range(length):
    vecList=[]
    sList=[]
    for j in range(K):
        p=int(Neighbors[i][j])
        v=location[p]-location[i]
        vecList.append(v)
        d=np.linalg.norm(v)
        sv=0
        if d!=0:
            for k in range(K):
                pos=int(Neighbors[i][k])
                v1=location[pos]-location[i]
                d1=np.linalg.norm(v1)
                if d1!=0:
                    v2=v*v1
                    s=0
                    for l in range(len(v2)):
                        s=s+v2[l]
                    cos=s/(d*d1)
                    if cos>=0.5:
                        sv=sv+cos
        sList.append(sv)
    ind=0
    maxV=0
    for j in range(len(sList)):
        if sList[j]>maxV:
            maxV=sList[j]
            ind=j
    vector[i]=vecList[ind]   

degree=input("Please input the angle value of the maximum deviation angle:")
degree=int(degree)
rv=math.radians(degree)
cv=math.cos(rv)

for i in range(length):
    v1=vector[i]
    l1=np.linalg.norm(v1)
    for j in range(len(Neighbors[i])):
        p=int(Neighbors[i][j])
        v2=location[p]-location[i]
        l2=np.linalg.norm(v2)
        v=v1*v2
        s=0
        for l in range(len(v)):
            s=s+v[l]
        if s!=0:
            cos=s/(l1*l2)
        else:
            cos=0
        if cos>=cv:
            Receivers[p].append(i)
            
result=np.ones(length,dtype=np.int)*(-1)
num=length
cno=0
while num>0:
    maxNum=-1
    npos=-1
    for i in range(length):
        if len(Receivers[i])>maxNum and result[i]==-1:
            maxNum=len(Receivers[i])
            npos=i
    start=npos
    if start==-1:
        break
    cno=cno+1
    result[start]=cno
    senderList=[]
    senderList.append(start)
    while len(senderList)>0:
        newSenderList=[]
        for i in range(len(senderList)):
            sender=senderList[i]
            for j in range(len(Receivers[sender])):
                receiver=Receivers[sender][j]
                if result[receiver]==-1:
                    result[receiver]=cno
                    newSenderList.append(receiver)
        senderList.clear()
        senderList=newSenderList
    num=0
    for i in range(length):
        if result[i]==-1:
            num=num+1
        
'''
preClass={}  
labelClass={}
statistics={}                  
for i in range(length):
    if result[i] not in preClass.keys():
        preClass[result[i]]=1
    else:
        preClass[result[i]]=preClass[result[i]]+1
    if label[i] not in labelClass.keys():
        labelClass[label[i]]=1
        statistics[label[i]]={}
    else:
        labelClass[label[i]]=labelClass[label[i]]+1

for i in range(length):
    if result[i] not in statistics[label[i]].keys():
        statistics[label[i]][result[i]]=1
    else:
        statistics[label[i]][result[i]]=statistics[label[i]][result[i]]+1

print(labelClass)
print(preClass)
print(statistics)
len1=0
len2=0
for i in range(length):
    if label[i]>len1:
        len1=label[i]
    if result[i]>len2:
        len2=result[i]

if len1>len2:
    maxL=len1
else:
    maxL=len2
sta=[]
for i in range(maxL+1):
    sta.append([])
    for j in range(maxL+1):
        sta[i].append(0)
        
print(len(label))

for i in range(len(label)):
    sta[label[i]][result[i]]=statistics[label[i]][result[i]]

res = hungalg.maximize(sta)
acc=0
for i in range(len(res)):
    acc=acc+sta[res[i][0]][res[i][1]]
acc=float(acc/length)
nmi=metrics.normalized_mutual_info_score(label,result)

print(acc)
print(nmi)


mds = manifold.MDS(max_iter=500, eps=1e-4, n_init=1,
                           dissimilarity='precomputed')
dp_mds = mds.fit_transform(dist)


R = list(range(256))
random.shuffle(R)
R = np.array(R)/255.0
G = list(range(256))
random.shuffle(G)
G = np.array(G)/255.0
B = list(range(256))
random.shuffle(B)
B = np.array(B)/255.0
colors = []
for i in range(256):
    colors.append((R[i], G[i], B[i]))

plt.figure(1)
for i in range(length):
    index = result[i]
    plt.plot(dp_mds[i][0], dp_mds[i][1], color = colors[index], marker = '.')
##plt.xlabel('x'), plt.ylabel('y')
plt.xticks([])  
plt.yticks([]) 
plt.show()

plt.figure(1)
for i in range(length):
    index = label[i]
    plt.plot(dp_mds[i][0], dp_mds[i][1], color = colors[index], marker = '.')
##plt.xlabel('x'), plt.ylabel('y')
plt.xticks([])  
plt.yticks([]) 
plt.show()



colors=['black','red','green','blue','orange','magenta','tomato','brown','darkslategrey','aquamarine','yellow','chocolate','khaki','gray','darkred','cyan','lightgreen','darkgoldenrod','cornflowerblue','springgreen','olive','darkcyan','sandybrown','orangered','lightcoral','violet','indianred','purple','lime','darkkhaki','dodgerblue','deeppink']
plt.figure(1)
for i in range(length):
    index = result[i]
    plt.plot(location[i][0], location[i][1], color = colors[index], marker = '.')
##plt.xlabel('x'), plt.ylabel('y')
plt.xticks([]) 
plt.yticks([]) 
plt.show()

'''
colors=['black','red','green','blue','orange','magenta','tomato','brown','purple','yellow','lightblue','chocolate','khaki','olivedrab','lightsteelblue','gray','cyan','orchid','palegreen','lightcoral','olive','lime','sandybrown','gold','springgreen','aquamarine','darkslategrey','violet','darkred','darkkhaki','pink','indianred']

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(length):
    index = result[i]
    ax.scatter(location[i][0], location[i][1], location[i][2], c=colors[index], marker='.')

 
plt.show()

               