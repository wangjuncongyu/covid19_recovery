# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:41:30 2020

@author: wjcongyu
"""
import _init_pathes
import os
import tensorflow as tf
from configs.cfgs import cfg
from models.risk_predictor import build_network
from data_process.data_generator1 import DataGenerator 
from data_process.data_processor import readCsv 
from models.backend import find_weights_of_last
import numpy as np
import argparse
import os.path as osp
from tensorflow.keras import models as KM
import pandas as pd
import datetime
parser = argparse.ArgumentParser()
parser.add_argument('-train_subsets', '--train_subsets', help='the subsets for training.', type = str, default ='1;2;3')
parser.add_argument('-eval_subsets', '--eval_subsets', help='the subset for test, others for training.', type = str, default ='4.lbl')
parser.add_argument('-batch_size', '--batch_size', help='the mini-batch size.', type = int, default = 72)
parser.add_argument('-cuda_device', '--cuda_device', help='runining on specified gpu', type = int, default = 0)
parser.add_argument('-save_root', '--save_root', help='root path to save the prediction results.', type = str, default = 'eval_results')#1:yes 0:no
parser.add_argument('-save_flag', '--save_flag', help='flag for saving result file name.', type = str, default = '')#1:yes 0:no
parser.add_argument('-treatment', '--treatment', help='the treatment for prediction.', type = str, default ='0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
parser.add_argument('-gt_ctimages', '--gt_ctimages', help='using ctimages or not.', type = int, default = 0)#1:yes 0:no
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
   
    eval_files = []
    eval_subsets = args.eval_subsets.split(';')
    for i in range(len(eval_subsets)):
        eval_files.append(os.path.join(cfg.data_set, eval_subsets[i]))
   
    val_data_generator = DataGenerator(eval_files, cfg, train_mode=False)
    eval_sample_num = val_data_generator.load_dataset()
   
    treattype = {0:'WithoutTreatment', 1:'WithTreatment'}
    if args.treatment == '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':
        model_dir_name = args.train_subsets +'_modelweights_'+ treattype[0]
    else:        
        model_dir_name = args.train_subsets +'_modelweights_'+ treattype[1]
    if args.gt_ctimages==0:
        model_dir_name += '_Withoutimage'
        
    #model_dir_name = args.train_subsets +'_modelweights_'+ treattype[1]
    treatments = np.array(args.treatment.split(','), dtype=np.int32)
    treatment_names = {'1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1':'GT',
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':'NONE',
                       '1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':'MPN',
                       '0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':'CP', 
                       '0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':'OV',
                       '0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':'TB',
                       '0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0':'ABD',
                       '0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0':'RV',
                       '0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0':'XBJ', 
                       '0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0':'LQC', 
                       '0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0':'CPP', 
                       '0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0':'PPL', 
                       '0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0':'MFN', 
                       '0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0':'LFN',
                       '0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0':'LZD',
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0':'HPN',
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0':'IGN',
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0':'VC', 
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0':'ACN', 
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0':'ABX', 
                       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1':'HFNC'}
    model = build_network([ cfg.treatment_infosize, cfg.im_feedsize, cfg.patient_infosize], [cfg.time_range], False, name='risk_predictor')

    feature_idx = [19, 31]
    
    checkpoint_file = find_weights_of_last(os.path.join(cfg.CHECKPOINTS_ROOT, model_dir_name), 'risk_predictor')
    print('############################',os.path.join(cfg.CHECKPOINTS_ROOT, model_dir_name))
    if checkpoint_file != '':  
        print ('@@@@@@@@@@ loading pretrained from ', checkpoint_file)
        model.load_weights(checkpoint_file)
    else:
        assert('no weight file found!!!')
   
    print (model.summary())
    
    #print layer information
    for i in range(len(model.layers)):
        layer = model.layers[i]      
        print(i, layer.name, layer.output.shape)
           
    #define the output of the network to get 
    outputs = [model.layers[i].output for i in feature_idx]
    pred_model = KM.Model(inputs=model.inputs, outputs=outputs)    
   
    save_root = args.save_root
    if not osp.exists(save_root):
        os.mkdir(save_root)
        
        
    feature_significance = []   
   
    risk_reg_preds = []
    gt_hitday = []
    gt_eventindicator = []
    gt_features = []
    gt_covid_severity = []
    gt_treatments = []
    gt_pids = []
    for step in range(eval_sample_num//(args.batch_size)+1):
        start = datetime.datetime.now()
        evbt_painfo, evbt_treatinfo, evbt_ims, evbt_treattimes,evbt_censor_indicator, evbt_severity, evbt_pids  \
                                                = val_data_generator.next_batch(args.batch_size)
        if args.treatment == '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1':
            feed_treatinfo = evbt_treatinfo            
        else:
            feed_treatinfo= np.zeros_like(evbt_treatinfo)
            feed_treatinfo[:,...] = treatments
            
        if args.gt_ctimages==0:
            feed_ims = tf.zeros_like(evbt_ims)
        else:
            feed_ims = evbt_ims
            
       
        feed = [feed_treatinfo, feed_ims, evbt_painfo]
            
        coff, reg_pred = pred_model(feed, training=False)
        end = datetime.datetime.now()
        print('processing time:', end-start)     
       
       
        risk_reg_preds.append(reg_pred)
        feature_significance.append(coff)
        gt_hitday.append(evbt_treattimes)
        gt_eventindicator.append(evbt_censor_indicator)
        gt_features.append(evbt_painfo)
        gt_covid_severity.append(evbt_severity)
        gt_treatments.append(evbt_treatinfo)
        gt_pids.append(evbt_pids)
            
    
    risk_reg_preds = np.concatenate(risk_reg_preds, axis=0)
    feature_significance = np.concatenate(feature_significance, axis=0)
    gt_hitday = np.concatenate(gt_hitday, axis=0)
    gt_eventindicator = np.concatenate(gt_eventindicator, axis=0)
    gt_features = np.concatenate(gt_features, axis=0)
    gt_covid_severity = np.concatenate(gt_covid_severity, axis=0)
    gt_treatments = np.concatenate(gt_treatments, axis=0)
    gt_pids = np.concatenate(gt_pids, axis=0)
    
    pinfo_header = readCsv(eval_files[0])[0][1:48]                
    pinfo_header = pinfo_header[0:2]+pinfo_header[3:]    
    
    csv_file = os.path.join(save_root, '{0}_{1}_risk_reg_preds.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(risk_reg_preds, columns=['day '+str(i) for i in range(cfg.time_range)])
    save_data.to_csv(csv_file,header=True, index=False)    
   
    
    csv_file = os.path.join(save_root, '{0}_{1}_gt_hitday.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(gt_hitday, columns=['hit day'])
    save_data.to_csv(csv_file,header=True, index=False)
    
    csv_file = os.path.join(save_root, '{0}_{1}_indicator.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(gt_eventindicator, columns=['indicator'])
    save_data.to_csv(csv_file,header=True, index=False)
    
    csv_file = os.path.join(save_root, '{0}_{1}_clinic_features.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(gt_features, columns=pinfo_header)
    save_data.to_csv(csv_file,header=True, index=False)
   
    csv_file = os.path.join(save_root, '{0}_{1}_gt_severity.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(gt_covid_severity, columns=['severity'])
    save_data.to_csv(csv_file,header=True, index=False)  
    
    csv_file = os.path.join(save_root, '{0}_{1}_gt_treatment.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    treat_header = ['MPN','CP','OV','TB','ABD','RV','XBJ','LQC','CPP','PPL','MFN','LFN','LZD','HPN','IGN','VC','ACN','ABX','HFNC']
    save_data = pd.DataFrame(gt_treatments, columns=treat_header)
    save_data.to_csv(csv_file,header=True, index=False)  
    
    csv_file = os.path.join(save_root, '{0}_{1}_patient_ids.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(gt_pids, columns=['pid'])
    save_data.to_csv(csv_file,header=True, index=False)  
   
       
    #pinfo_header.append('cnn_feature')
    csv_file = os.path.join(save_root, '{0}_{1}_feature_significance.csv'.format(args.eval_subsets+args.save_flag, treatment_names[args.treatment]))
    save_data = pd.DataFrame(feature_significance, columns=pinfo_header)
    save_data.to_csv(csv_file,header=True, index=False)
    
  

if __name__ == "__main__":
    args = parser.parse_args()
    print (args)
    
    main(args)
