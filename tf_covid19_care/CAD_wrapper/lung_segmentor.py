# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:13:25 2019

@author: wjcongyu
"""
from models.lungseg_3dunet import Seg_3DUnet
from data_process.data_processor import resize
from data_process.data_processor import hu2gray
from data_process.data_processor import get_chest_roi
from data_process.data_processor import keep_max_object
from data_process.data_processor import get_convex_bbox_frm_3dmask
from scipy import ndimage
import numpy as np
import os.path as osp
class LungSegmentor(object):
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
        model_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, 'lung_3dseg')
        assert osp.exists(model_dir), 'no model dir found:' + model_dir        
        try:        
            self.model = Seg_3DUnet(self.cfg.INPUT_SHAPE, is_training=False, config= self.cfg, num_classes = 3, model_dir = model_dir)
            checkpoint = self.model.find_last()
            print (checkpoint)
            assert osp.exists(checkpoint), 'no checkpoint found:' + checkpoint
            self.model.load_weights(checkpoint)
            self.is_ready = True
            return True                  
                  
        except (OSError, TypeError) as reason:
            self._record_error(str(reason))
            self.is_ready = False
            return False
          
            
    def segment_lung(self, volume):
        self.last_error = ''
        if not self.is_ready:
            self._record_error('model not ready for lobe segmentation!')
            return np.zeros_like(volume)
        
        if volume is None:
            self._record_error('None volume data not allowed!')
            return None
        
        if len(volume.shape)!=3:
            self._record_error('volume shape must equals 3! depthxheights x widths')
            return np.zeros_like(volume)
      
        roi_bbox, lung = get_chest_roi(volume)
        x1,y1,z1,x2,y2,z2 = roi_bbox
        if x2-x1<30 or y2-y1<30 or z2-z1<20:
            self._record_error('chest roi segmentation failed!')
            return np.zeros_like(volume)
        roi_volume = volume.copy()[z1:z2,y1:y2,x1:x2]
        lung = lung[z1:z2,y1:y2,x1:x2]
        im_data =resize(roi_volume, self.cfg.INPUT_SHAPE)
        #im_data = hu2gray(im_data, WL=-500, WW=1500)
        im_data = im_data.reshape((1,im_data.shape[0],im_data.shape[1],im_data.shape[2],1))
       
        lung_preds = self.model.predict(im_data)
        
        lung_pred_labels = np.argmax(lung_preds[0,:,:,:,:],axis=-1)

        pred_mask = np.uint8(lung_pred_labels)  
       
        roi_mask = np.zeros(roi_volume.shape,dtype=np.uint8)
        
        for k in range(1, 3):
            k_mask = np.zeros_like(pred_mask, dtype=np.uint8)
            k_mask[pred_mask==k] = 1
            k_mask = np.uint8(np.ceil(resize(k_mask,roi_volume.shape)))
            k_mask = keep_max_object(k_mask)
            k_mask = ndimage.binary_closing(k_mask,structure=np.ones((7,7,7)))
            roi_mask[k_mask>=0.4] = k
                
        final_mask = np.zeros_like(volume, dtype = np.uint8)
        
        final_mask[z1:z2,y1:y2, x1:x2] = roi_mask      
        
       
        return final_mask
        
            
    def _record_error(self, error):
        self.last_error = error