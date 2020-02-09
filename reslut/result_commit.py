# #encoding:utf/8
# import sys
# from mmdet.apis import inference_detector, init_detector
# import json
# import cv2
# import os
# import numpy as np
# import argparse
# from tqdm import tqdm
# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)
#
# #generate result
# def result_from_dir():
# 	index = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
# 	# build the model from a config file and a checkpoint file
# 	model = init_detector(config2make_json, model2make_json, device='cuda:0')
# 	pics = os.listdir(pic_path)
# 	meta = {}
# 	images = []
# 	annotations = []
# 	num = 0
# 	for im in tqdm(pics):
# 		num += 1
# 		img = os.path.join(pic_path,im)
# 		result_ = inference_detector(model, img)
# 		images_anno = {}
# 		images_anno['file_name'] = im
# 		images_anno['id'] = int(num)
# 		images.append(images_anno)
# 		for i ,boxes in enumerate(result_,1):
# 			if len(boxes):
# 				defect_label = index[i]
# 				for box in boxes:
# 					anno = {}
# 					anno['image_id'] =int( num)
# 					anno['category_id'] = defect_label
# 					anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
# 					anno['bbox'][2] = anno['bbox'][2]-anno['bbox'][0]
# 					anno['bbox'][3] = anno['bbox'][3]-anno['bbox'][1]
# 					anno['score'] = float(box[4])
# 					if(anno['score'] <0.005):
# 						anno['category_id'] = 0
# 						continue
# 					annotations.append(anno)
# 					#draw bbox
# 					img_temp=cv2.imread(img)
# 					draw_0=cv2.rectangle(img_temp, (int(anno['bbox'][0]),int(anno['bbox'][1])), (int(anno['bbox'][0])+int(anno['bbox'][2]),int(anno['bbox'][1])+
# 					int(anno['bbox'][3])), (0,255,0))
# 					text=str(anno['category_id'])+'--'+str(anno['score'])
# 					cv2.putText(draw_0, text, (int(anno['bbox'][0]),int(anno['bbox'][1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
# 					cv2.imshow("draw_0", draw_0)#显示画过矩形框的图片
# 					cv2.waitKey(0)
# 					cv2.destroyWindow("draw_0")
#
#     #CLASSES = ('瓶盖破损', '瓶盖变形', '瓶盖坏边', '瓶盖打旋',  '瓶盖断点', '标贴歪斜', '标贴起皱',  '标贴气泡', '喷码正常', '喷码异常')
# 	meta['images'] = images
# 	meta['annotations'] = annotations
#
# 	# with open(json_out_path, 'w') as fp:
# 	# 	json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))
# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(description="Generate result")
# 	parser.add_argument("-m", "--model",default='/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/202002041723_pg/latest.pth',help="Model path",type=str,)
# 	parser.add_argument("-c", "--config",default='/home/liusiyu/liuxin/mmdetection/my_configs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py',help="Config path",type=str,)
# 	parser.add_argument("-im", "--im_dir",default='/home/liusiyu/liuxin/mmdetection/data/coco/testimages/',help="Image path",type=str,)
# 	parser.add_argument('-o', "--out",default='/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/202002041723_pg/reslut.json',help="Save path", type=str,)
# 	args = parser.parse_args()
# 	model2make_json = args.model
#
# 	config2make_json = args.config
# 	json_out_path = args.out
# 	pic_path = args.im_dir
# 	result_from_dir()



# #encoding:utf/8
import sys
from mmdet.apis import inference_detector, init_detector
import json
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class COCO_Set():
    '''
    des:
    '''
    # 定义类变量，完全不变
    categories = [{"supercategory": "\u74f6\u76d6\u7834\u635f", "id": 1, "name": "\u74f6\u76d6\u7834\u635f"},
                  {"supercategory": "\u55b7\u7801\u6b63\u5e38", "id": 9, "name": "\u55b7\u7801\u6b63\u5e38"},
                  {"supercategory": "\u74f6\u76d6\u65ad\u70b9", "id": 5, "name": "\u74f6\u76d6\u65ad\u70b9"},
                  {"supercategory": "\u74f6\u76d6\u574f\u8fb9", "id": 3, "name": "\u74f6\u76d6\u574f\u8fb9"},
                  {"supercategory": "\u74f6\u76d6\u6253\u65cb", "id": 4, "name": "\u74f6\u76d6\ \u65cb"},
                  {"supercategory": "\u80cc\u666f", "id": 0, "name": "\u80cc\u666f"},
                  {"supercategory": "\u74f6\u76d6\u53d8\u5f62", "id": 2, "name": "\u74f6\u76d6\u53d8\u5f62"},
                  {"supercategory": "\u6807\u8d34\u6c14\u6ce1", "id": 8, "name": "\u6807\u8d34\u6c14\u6ce1"},
                  {"supercategory": "\u6807\u8d34\u6b6a\u659c", "id": 6, "name": "\u6807\u8d34\u6b6a\u659c"},
                  {"supercategory": "\u55b7\u7801\u5f02\u5e38", "id": 10, "name": "\u55b7\u7801\u5f02\u5e38"},
                  {"supercategory": "\u6807\u8d34\u8d77\u76b1", "id": 7, "name": "\u6807\u8d34\u8d77\u76b1"}]

    def __init__(self, name, images_path='./images/'):
        # 初始化coco数据集基本信息
        self.name = name
        self.images_pth = images_path
        self.info = []
        self.images = []
        self.license = []
        self.annotations = []
        self.json_file = None

    # 转化json对象为python对象
    def load_json(self,json_path):
        with  open(json_path)  as f:
            self.json_file = json.load(f)  # 读入到python对象
            return self.json_file
        return None

    # 获取json values
    def get_items(self, key):
        return self.json_file[key]

    # 写入json
    def write_file(self, json_name):
        self.write_json = {'info': self.info, 'images': self.images, 'license': self.license,
                           'categories': self.categories, 'annotations': self.annotations}
        with open(json_name, "w") as f:
            json.dump(self.write_json, f, indent=4)

def main():

    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("--test_input",default='/home/liusiyu/liuxin/mmdetection/data/coco/annotations/pgtest2017.json',help="test_json path", type=str,)
    parser.add_argument('--reslut_input', default=
    '/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/reslut.bbox.json',help="result_json path", type=str,)
    parser.add_argument('-o', "--out",default=
    '/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/reslut.json',help="Save path", type=str,)
    args = parser.parse_args()
    # 	model2make_json = args.model
    test_set = COCO_Set('test_json', images_path=None)
    reslut_set = COCO_Set('result_json', images_path=None)
    tejson_file=test_set.load_json(json_path=args.test_input)
    resjson_file=reslut_set.load_json(json_path=args.reslut_input)
    test_set.images=tejson_file['images']
    for index ,i in enumerate(resjson_file):
        if i["category_id"] == 6 or i["category_id"] == 7 or i["category_id"] == 8:
            if i["score"] > 0.005:
                score = i["score"]
                i["score"] = round(score, 4)
                test_set.annotations.append(i)
        else:
            if i["score"] > 0.005:
                score = i["score"]
                i["score"] = round(score, 4)
                test_set.annotations.append(i)

    test_set.write_file(json_name=args.out)



if __name__ == "__main__":
    main()

