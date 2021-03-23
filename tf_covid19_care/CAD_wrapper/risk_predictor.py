# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:13:25 2019

@author: wjcongyu
"""
from models.risk_predictor import build_network
from models.backend import find_weights_of_last
from tensorflow.keras import models as KM
import numpy as np
import os.path as osp

class RiskPredictor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_error = ''
        self.is_ready = False
        self.sess = None
        self.model = None
        
    def initialize(self):
        '''
        Create session and Loading models and weights from disk. You have to call this func before
        calling classify
        '''
        model_withouttreatment_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, '1;2;3;4;5_modelweights_WithoutTreatment')
        assert osp.exists(model_withouttreatment_dir), 'no model dir found:' + model_withouttreatment_dir  
        
        model_withtreatment_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, '1;2;3;4;5_modelweights_WithTreatment')
        assert osp.exists(model_withtreatment_dir), 'no model dir found:' + model_withtreatment_dir  
        try:        
            self.model_withouttreat = build_network([self.cfg.treatment_infosize, self.cfg.im_feedsize, self.cfg.patient_infosize], [self.cfg.severity_categories, self.cfg.time_range], False, name='risk_predictor')
            checkpoint_file = find_weights_of_last(model_withouttreatment_dir, 'risk_predictor')            
            print (checkpoint_file)
            assert osp.exists(checkpoint_file), 'no checkpoint found:' + checkpoint_file
            self.model_withouttreat.load_weights(checkpoint_file)
            
            self.model_withtreat = build_network([self.cfg.treatment_infosize, self.cfg.im_feedsize, self.cfg.patient_infosize], [self.cfg.severity_categories, self.cfg.time_range], False, name='risk_predictor')
            checkpoint_file = find_weights_of_last(model_withtreatment_dir, 'risk_predictor')            
            print (checkpoint_file)
            assert osp.exists(checkpoint_file), 'no checkpoint found:' + checkpoint_file
            self.model_withtreat.load_weights(checkpoint_file)
            
            feature_idx = [23, 34, 35]
            outputs = [self.model_withouttreat.layers[i].output for i in feature_idx]
            self.model_withouttreat = KM.Model(inputs=self.model_withouttreat.inputs, outputs=outputs)    
            
            outputs = [self.model_withtreat.layers[i].output for i in feature_idx]
            self.model_withtreat = KM.Model(inputs=self.model_withtreat.inputs, outputs=outputs)    
            self.is_ready = True
            return True                  
                  
        except (OSError, TypeError) as reason:
            self._record_error(str(reason))
            self.is_ready = False
            return False
          
            
    def predict(self, treatment_scheme, ct_scan, patient_info):
        self.last_error = ''
        if not self.is_ready:
            self._record_error('model not ready for risk prediction!')
            return None, None, None
        
        feed = [np.float32(treatment_scheme), np.float32(ct_scan), np.float32(patient_info)]
        
        if np.sum(treatment_scheme>0) == 0:
            coff, cls_pred, reg_pred = self.model_withouttreat(feed, training=False)
        else:
            print('##################')
            coff, cls_pred, reg_pred = self.model_withtreat(feed, training=False)
        return np.array(coff), np.array(cls_pred), np.array(reg_pred)
        
            
    def _record_error(self, error):
        self.last_error = error