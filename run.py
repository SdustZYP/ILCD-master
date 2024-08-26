# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.base_cfgs import Cfgs
from core.exec import Execution, loss_list
import argparse, yaml
from visdom import Visdom
import time
import cv2
import torch
import os
import plotly as plt
import random
import numpy as np
# from core.model.net import vs

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=['small', 'large'],
                      help='{small, large}',
                      default='small', type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",
                      type=bool)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      help='set True to save the '
                           'prediction vectors'
                           '(only work in testing)',
                      type=bool)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size during training',
                      type=int)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                      help='max training epoch',
                      type=int)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                      help='pre-load the features into memory'
                           'to increase the I/O speed',
                      type=bool)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu select, eg.'0, 1, 2'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      help='resume training',
                      type=bool)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      help='use pin memory',
                      type=bool)

    parser.add_argument('--VERB', dest='VERBOSE',
                      help='verbose print',
                      type=bool)

    parser.add_argument('--DATA_PATH', dest='DATASET_PATH',
                      help='vqav2 dataset root path',
                      type=str)

    parser.add_argument('--FEAT_PATH', dest='FEATURE_PATH',
                      help='bottom up features root path',
                      type=str)

    args = parser.parse_args()
    return args

def random_num(size,end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls



if __name__ == '__main__':
    __C = Cfgs()

    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)

    __C.check_path()

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)

    # # 训练过程的loss显示
    # viz = Visdom()
    # # print(len(loss_list))
    # for i in range(3):
    #     loss_list.remove(loss_list[len(loss_list) - 1])
    # # print(len(loss_list))
    # viz.line([0.], [0.], win='train', opts=dict(title='loss', legend=['loss']))
    # for globals_steps in range(len(loss_list)):
    #     loss = loss_list[globals_steps]
    #     viz.line([[loss]], [globals_steps], win='train', update='append')
    #     time.sleep(0.5)

    # # 显示attention的热力图
    # v = vs[0][0]
    # v = v.data.squeeze(0)
    # v = v.data.cpu().numpy()
    # channel_num = random_num(25, v.shape[0])
    # plt.figure(figsize=(10, 10))
    # for index, channel in enumerate(channel_num):
    #     ax = plt.subplot(5, 5, index + 1)
    #     plt.imshow(v[channel, :])
    # plt.savefig("D:/faster_rcnn/save-1.jpg", dpi=300)
    #
    # img_path = "D:/jpgs_of_vqa/jpgs/069/6216.jpg"
    # features = vs
    #
    # features.retain_grad()
    # t = model.avgpool(features)
    # t = t.reshape(1, -1)
    # output = model.classifier(t)[0]
    # pred = torch.argmax(output).item()
    # pred_class = output[pred]
    #
    # pred_class.backward()
    #
    # grads = features.grad
    #
    # features = features[0][0][0]
    # avg_grads = torch.mean(grads[0], dim=(1, 2))
    # avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
    # features *= avg_grads
    # features = features * 255.0
    #
    # heatmap = features.detach().cpu().numpy()
    # heatmap = np.mean(heatmap, axis=0)
    #
    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= (np.max(heatmap) + 1e-8)
    #
    # img = cv2.imread(img_path)
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    # cv2.imshow('1', superimposed_img)
    # cv2.waitKey(0)

