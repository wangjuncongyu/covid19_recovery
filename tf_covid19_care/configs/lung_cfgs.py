# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:00:27 2019

@author: wjcongyu
"""
from easydict import EasyDict as edict

cfg = edict()
cfg.CHECKPOINTS_ROOT = '../checkpoints'
cfg.INPUT_SHAPE = [128, 96, 128] #[D, H, W]
cfg.MOMENTUM = 0.09
cfg.STEPS_PER_EPOCH = 100
cfg.MAX_KEEPS_CHECKPOINTS = 1