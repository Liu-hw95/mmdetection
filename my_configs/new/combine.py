import json
import os
import cv2
with open(r"D:\chongqing1_round1_testA_20191223\pgtest2017.json",'r') as load_f:
    f1= json.load(load_f)
with open(r"D:\chongqing1_round1_testA_20191223\ptest2017.json",'r') as load_f:
    f2= json.load(load_f)
b1=f1["images"]
b2=f2["images"]
a=[]
for i in b1:
    del i["width"]
    del i["height"]
    a.append(i)
for j in b2:
    del j["width"]
    del j["height"]
    a.append(j)

with open(r"D:\chongqing1_round1_testA_20191223\results.bbox.json",'r') as load_f:
    f= json.load(load_f)
with open(r"D:\chongqing1_round1_testA_20191223\presults.bbox.json",'r') as load_f:
    p = json.load(load_f)
f1=f
f2=p
for n, i in enumerate(f1):
    if i["category_id"] == 6:
        f[n]["category_id"] = 9
    if i["category_id"] == 7:
        f[n]["category_id"] =10
for n,i in enumerate(f2):
    if i["category_id"] == 1:
        p[n]["category_id"]=6
    if i["category_id"] == 2:
        p[n]["category_id"]=7
    if i["category_id"] == 3:
        p[n]["category_id"]=8
for i in p :
    f.append(i)
for i,m in enumerate(f):
    for j,n in enumerate(m["bbox"]):
        f[i]["bbox"][j]=round(n,2)
        # f[i]["bbox"][j]=int(n)
test={
    "images":a,
    "annotations": f,
}

with open(r"D:\chongqing1_round1_testA_20191223\sum多尺度多ratio.json",'w') as fp:
    json.dump(test, fp)