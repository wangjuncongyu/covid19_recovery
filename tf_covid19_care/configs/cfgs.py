# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:00:27 2019

@author: wjcongyu
"""
from easydict import EasyDict as edict

cfg = edict()
cfg.data_set = r'D:\data\tf_pneumonia_db'
cfg.CHECKPOINTS_ROOT = '../checkpoints'
cfg.im_feedsize = [48,48,48]
cfg.im_path = r'D:\data\tf_pneumonia_db\ct_images_lungonly-48'

cfg.patient_infosize = 46 #[D, H, W]
cfg.treatment_infosize = 19
cfg.time_range = 32
cfg.MOMENTUM = 0.09
cfg.MAX_KEEPS_CHECKPOINTS = 1