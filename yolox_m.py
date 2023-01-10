#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (416, 416)
        self.test_size = (416, 416)

        # Define yourself dataset path
        self.data_dir = "/content/dataset/images"
        self.train_ann = "/content/dataset/annotations/train_annotations.json"
        self.val_ann = "/content/dataset/annotations/validation_annotations.json"

        self.num_classes = 1

        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 1