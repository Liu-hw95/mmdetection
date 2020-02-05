import json
import numpy as np
with open(r"/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/202002041723_pg/reslut.bbox.json",'r') as load_f:
    f = json.load(load_f)
with open(r"/home/liusiyu/liuxin/mmdetection/data/src/annotations.json",'r') as load_f:
    fp = json.load(load_f)
print(type(f["annotations"]))
fa=f["annotations"]
fp=fp["annotations"]
name=f["images"]
thr=0.1
f=fa
a=set()
b1=set()
e=[]
m=[]
scoremax=0
count=0
count1=0
count2=0
c=np.ones(10)*500
c1=np.zeros(10)
d=np.ones(10)*500
d1=np.zeros(10)
e1=np.ones(10)*500
e2=np.zeros(10)
f1=np.ones(10)*500
f2=np.zeros(10)
# for i in f:
#     if i["category_id"] not in a:
#         a.add(i["category_id"])
#     b=int(i["category_id"])
#     c[b-1]=min(c[b-1],i["bbox"][2])
#     c1[b - 1] = max(c1[b - 1], i["bbox"][2])
# for j in fp:
#     b=int(j["category_id"])
#     d[b-1]=min(d[b-1],j["bbox"][2])
#     d1[b - 1] = max(d1[b - 1], j["bbox"][2])
# for i in f:
#     if i["category_id"] not in b1:
#         b1.add(i["category_id"])
#     b=int(i["category_id"])
#     e1[b-1]=min(e1[b-1],i["bbox"][3])
#     e2[b - 1] = max(e2[b - 1], i["bbox"][3])
# for j in fp:
#     b=int(j["category_id"])
#     f1[b-1]=min(f1[b-1],j["bbox"][3])
#     f2[b - 1] = max(f2[b - 1], j["bbox"][3])
for n,i in enumerate(f):
    # b = int(i["category_id"])
    # if i["bbox"][2] < d[b-1]*(0.4) or i["bbox"][3] < f1[b-1]*(0.4) or i["bbox"][2] > d1[b-1]*(2) or i["bbox"][3] > f2[b-1]*(2):
    #      count = count + 1
    # if(i["score"]>0.0000015):
    #     b=i["image_id"]
    #     if b==f[n-1]["image_id"]:
    #         m.append(n)
    #         scoremax=max(scoremax,i["score"])
    #         if n ==len(f)-1 and scoremax > thr:
    #             for j in m:
    #                 f[j]["score"]=round(f[j]["score"], 4)
    #                 e.append(f[j])
    #     elif scoremax <thr:
    #         print(scoremax)
    #         m=[]
    #         scoremax=0
    #     else:
    #         for j in m:
    #             f[j]["score"] = round(f[j]["score"], 4)
    #             e.append(f[j])
    #         m=[]
    #         scoremax=0
    if i["category_id"] == 6 or i["category_id"] == 7 or i["category_id"] == 8:
        if i["score"]>0.05:
            score=i["score"]
            i["score"] = round(score, 4)
            e.append(i)
    else:
        if i["score"]>0.000286:
            score=i["score"]
            i["score"] = round(score, 4)
            e.append(i)




test={
    "images":name,
    "annotations": e,
}
with open(r"/home/liusiyu/liuxin/mmdetection/reslut.json",'w') as fp:
    json.dump(test, fp)
print(a)
print(c)
print(d)
print(count)
print(count1)
print(count2)
print(e1)
print(f1)