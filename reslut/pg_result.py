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
# 	# index = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
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
# 				defect_label = i
# 				for box in boxes:
# 					anno = {}
# 					anno['image_id'] =int( num)
# 					anno['category_id'] = defect_label
# 					anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
# 					anno['bbox'][2] = anno['bbox'][2]-anno['bbox'][0]
# 					anno['bbox'][3] = anno['bbox'][3]-anno['bbox'][1]
# 					anno['score'] = float(box[4])
# 					# if(anno['score'] <0.08):
# 					# 	anno['category_id'] = 0
# 					# 	continue
# 					annotations.append(anno)
# 					#draw bbox
# 					# img_temp=cv2.imread(img)
# 					# draw_0=cv2.rectangle(img_temp, (int(anno['bbox'][0]),int(anno['bbox'][1])), (int(anno['bbox'][0])+int(anno['bbox'][2]),int(anno['bbox'][1])+
# 					# int(anno['bbox'][3])), (0,255,0))
# 					# text=str(anno['category_id'])+'--'+str(anno['score'])
# 					# cv2.putText(draw_0, text, (int(anno['bbox'][0]),int(anno['bbox'][1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
# 					# cv2.imshow("draw_0", draw_0)#显示画过矩形框的图片
# 					# cv2.waitKey(0)
# 					# cv2.destroyWindow("draw_0")
#
#     #CLASSES = ('瓶盖破损', '瓶盖变形', '瓶盖坏边', '瓶盖打旋',  '瓶盖断点', '标贴歪斜', '标贴起皱',  '标贴气泡', '喷码正常', '喷码异常')
# 	meta['images'] = images
# 	meta['annotations'] = annotations
#
# 	with open(json_out_path, 'w') as fp:
# 		json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))
# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(description="Generate result")
# 	parser.add_argument("-m", "--model",default='/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/202002031943_pg/epoch_19.pth',help="Model path",type=str,)
# 	parser.add_argument("-c", "--config",default='/home/liusiyu/liuxin/mmdetection/my_configs/pg_baseline.py',help="Config path",type=str,)
# 	parser.add_argument("-im", "--im_dir",default='/home/liusiyu/liuxin/mmdetection/data/coco/images/',help="Image path",type=str,)
# 	parser.add_argument('-o', "--out",default='/home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/202002031943_pg/reslut.json',help="Save path", type=str,)
# 	args = parser.parse_args()
# 	model2make_json = args.model
#
# 	config2make_json = args.config
# 	json_out_path = args.out
# 	pic_path = args.im_dir
# 	result_from_dir()


#tools/dist_test.sh /home/liusiyu/liuxin/mmdetection/my_configs/cascade_rcnn_r50_fpn_1x_78.py /home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x/202002051610/latest.pth 4 --json_out /home/liusiyu/liuxin/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x/202002051610/reslut.json