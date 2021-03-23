# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:28:04 2019

@author: wjcongyu
"""

import numpy as np
import os.path as osp
import pandas as pd
from .data_processor import resize, hu2gray


class DataGenerator(object):
    '''
    this class defines functions that used to prepare data for trarining and evaluation
    '''
    def __init__(self, anno_files, cfg, train_mode=True):
        assert len(anno_files) > 0, 'no annotation file!'
        
        self.annotation_files = anno_files
        self.cfg = cfg
        self.train_mode = train_mode
        self.data_ready = False
        
    def load_dataset(self):    
        '''
        After instantiation of this class, call this function to load data from the csv file
        '''
        self.patients=[]
        self.total = 0
        self.datset_root = ''
        for anno_file in self.annotation_files:
            self.datset_root = osp.dirname(anno_file)
            print ('loading from:', anno_file)
            pd_data = pd.read_csv(anno_file)
            subset = np.array(pd_data.iloc[0:])
            self.patients.append(subset)
            self.header = pd_data.head()
            
        self.patients = np.concatenate(self.patients, axis=0)  
        '''if self.train_mode:
            days = self.patients[...,69]
            keep = np.where(np.where(days>=20, 1, 0) * np.where(days<=30, 1, 0)==1)
            extent_patients = self.patients[keep]
            print('extending patients [20 30):', extent_patients.shape[0])
            self.patients = np.concatenate([self.patients, extent_patients], axis=0)'''
       
        self.total = self.patients.shape[0]
       
        self.samples_idx = [i for i in range(self.total)]
        if osp.exists('../checkpoints/feature_minv.npy'):
            print('found minv file for normalization')
            self.patient_infominv = np.load('../checkpoints/feature_minv.npy', allow_pickle=True)
            print(self.patient_infominv)
        else:
            self.patient_infominv = np.min(self.patients[:, 1:48], axis=0)
        self.patient_infominv[1] = 1
        
        if osp.exists('../checkpoints/feature_maxv.npy'):
            print('found maxv file for normalization')
            self.patient_infomaxv = np.load('../checkpoints/feature_maxv.npy', allow_pickle=True)
            print(self.patient_infomaxv)
        else:
            self.patient_infomaxv = np.max(self.patients[:, 1:48], axis=0)
        self.patient_infomaxv[1] = 120
        
        
        np.random.shuffle(self.samples_idx)
        self.current_idx = 0
        return self.total    
  
    def next_batch(self, batch_size): 
        '''
        cals this fuction to get a mini-batch that can be fed to the model
        '''
        if batch_size>self.total:
            batch_size = self.total
            
        bt_pids = []
        bt_painfo = []
        bt_severity = []
        bt_treatment_scheme = []
        bt_treatment_days = []     
        bt_event_indicator = []
        bt_ims = []
        for k in range(batch_size):         
            current_patient_idx = self.samples_idx[int((self.current_idx + k) % len(self.samples_idx))]
            patient= self.patients[current_patient_idx, ...]
            if self.train_mode==False and self.current_idx+k>=len(self.samples_idx):
                continue
            
            patient_info = patient[1:48]
            #normalize the information
            patient_info = (patient_info-self.patient_infominv)/(self.patient_infomaxv-self.patient_infominv+10e-5)
            bt_severity.append(patient[3])
            
            #do not feed the covid-19 severity to the network
            patient_info = np.delete(patient_info, 2, axis=0)
            patient_info[patient_info<0] = 0
            patient_info[patient_info>1.0] = 1.0
            
            
            bt_pids.append(patient[0])
            bt_painfo.append(patient_info)
            bt_treatment_scheme.append(patient[48:67])
            bt_treatment_days.append(patient[67])
            bt_event_indicator.append(patient[68])
            im_file = osp.join(self.datset_root, 'ct_images', patient[69])
            
            if osp.exists(im_file): 
                #im_data = resize(hu2gray(np.load(im_file),WL=-500, WW=1200), self.cfg.im_feedsize)
                im_data = np.load(im_file)
                if self.train_mode:
                    im_data = self.__augment(im_data, self.cfg)
                im_data = im_data.reshape(im_data.shape + (1,))
                bt_ims.append(im_data)
            else:
                im_data = np.zeros(self.cfg.im_feedsize+ [1], dtype=np.float32)               
                bt_ims.append(im_data)
                
        bt_pids = np.array(bt_pids)
        bt_painfo = np.array(bt_painfo,dtype=np.float32)       
        bt_treatment_scheme = np.array(bt_treatment_scheme,dtype=np.float32)
        bt_treatment_days = np.array(bt_treatment_days,dtype=np.int32)
        bt_treatment_days[bt_treatment_days>=self.cfg.time_range-2] = self.cfg.time_range-2
     
        bt_ims = np.array(bt_ims,dtype=np.float32)
        bt_event_indicator = np.array(bt_event_indicator,dtype=np.int32)
        keep_death = np.where(bt_event_indicator==2)
        bt_treatment_days[keep_death] = self.cfg.time_range-1
        bt_severity = np.array(bt_severity,dtype=np.int32)        
     
        #sorting according to treatment time
        sort_idx = np.argsort(bt_treatment_days)
        bt_pids = bt_pids[sort_idx]
        bt_painfo = bt_painfo[sort_idx]
        bt_treatment_scheme = bt_treatment_scheme[sort_idx]
        bt_ims = bt_ims[sort_idx]
        bt_treatment_days = bt_treatment_days[sort_idx]
        bt_event_indicator = bt_event_indicator[sort_idx]     
        bt_severity = bt_severity[sort_idx]
        
        #shuffle the samples
        self.current_idx += (batch_size)
        if self.current_idx >= len(self.samples_idx):
            self.current_idx = 0
            np.random.shuffle(self.samples_idx)            
       
        return bt_painfo, bt_treatment_scheme, bt_ims, bt_treatment_days, bt_event_indicator, bt_severity, bt_pids
  
    
    #=========== private function of augmnet CT images======================
    def __augment(self,im,cfg):
        im = self.__random_rot(im)
        im = self._random_flip(im)               
        return np.float32(im)
    
    def __random_rot(self, im):
        rot = np.random.choice([0, 90, 180, 270], size=1)[0]
        k = int(rot/90)
        if k == 0:
            return im
        
        axis = np.random.choice([0, 1, 2],size=1)
        if axis == 1:
            im = im.transpose((1, 0, 2))
            im = np.rot90(im,k=k)
            im = im.transpose((1, 0, 2))
        elif axis == 2:
            im = im.transpose((0, 2, 1))
            im = np.rot90(im,k=k)
            im = im.transpose((0, 2, 1))
        else:
            im = np.rot90(im,k=k)
        return im
    
    def _random_flip(self, im):
        flip = np.random.choice([0, 1, 2, 3],size=1)[0]
           
        if flip>0:
            im = np.flip(im,axis = flip-1)
        return im
        
