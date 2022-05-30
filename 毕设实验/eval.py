# -*- coding: utf-8 -*-

import numpy as np
import json

import paddle

from reader import test_data_loader
from multinms import multiclass_nms
from yolov3_spp import YOLOv3
#from yolov3 import YOLOv3

import argparse

def parse_args():
    parser = argparse.ArgumentParser("Evaluation Parameters")
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./insects/test/images',
        help='the directory of test images')
    parser.add_argument(
        '--weight_file',
        type=str,
        default='./yolo_epoch50.pdparams',
        help='the path of model parameters')
    args = parser.parse_args()
    return args


args = parse_args()
TESTDIR = args.image_dir
WEIGHT_FILE = args.weight_file

#ANCHORS = [22, 53, 30, 52, 27, 60, 33, 57, 24, 84, 37, 71, 43, 78, 53, 81, 62, 98]
ANCHORS = [23, 48, 32, 42, 23, 84, 32, 62, 51, 64, 27, 124, 41, 84, 63, 102, 72, 225]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

VALID_THRESH = 0.01

NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

NUM_CLASSES = 7

TESTDIR = './insects/val/images' #请将此目录修改成用户自己保存测试图片的路径
WEIGHT_FILE = './yolo_epoch220.pdparams' # 请将此文件名修改成用户自己训练好的权重参数存放路径
#TESTDIR = './val/images'

if __name__ == '__main__':
    paddle.set_device("cpu")
    model = YOLOv3(num_classes=NUM_CLASSES)
    params_file_path = WEIGHT_FILE
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    total_results = []
    test_loader = test_data_loader(TESTDIR, batch_size= 1, mode='test')
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale_data)

        outputs = model.forward(img)
        bboxes, scores = model.get_pred(outputs,
                                 im_shape=img_scale,
                                 anchors=ANCHORS,
                                 anchor_masks=ANCHOR_MASKS,
                                 valid_thresh = VALID_THRESH)

        bboxes_data = bboxes.numpy()
        scores_data = scores.numpy()
        result = multiclass_nms(bboxes_data, scores_data,
                      score_thresh=VALID_THRESH, 
                      nms_thresh=NMS_THRESH, 
                      pre_nms_topk=NMS_TOPK, 
                      pos_nms_topk=NMS_POSK)
        for j in range(len(result)):
            result_j = result[j]
            img_name_j = img_name[j]
            total_results.append([img_name_j, result_j.tolist()])
        print('processed {} pictures'.format(len(total_results)))

    print('')
    json.dump(total_results, open('pred_results.json', 'w'))

