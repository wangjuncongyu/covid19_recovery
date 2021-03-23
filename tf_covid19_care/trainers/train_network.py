# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:41:30 2020

@author: wjcongyu SJTU
"""
import _init_pathes
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as KB
from tensorflow.keras import losses as KLOSS
from configs.cfgs import cfg
from models.risk_predictor import build_network
from data_process.data_generator import DataGenerator 
from models.backend import find_weights_of_last, delete_old_weights, draw_progress_bar
import numpy as np
import argparse
import os.path as osp
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

parser = argparse.ArgumentParser()
parser.add_argument('-train_subsets', '--train_subsets', help='the subsets for training.', type = str, default ='1;2;3')
parser.add_argument('-eval_subsets', '--eval_subsets', help='the subset for test, others for training.', type = str, default ='4.lbl')
parser.add_argument('-load_pretrain', '--load_pretrain', help='1:initialize model with pretrained weights.', type = int, default = 1)
parser.add_argument('-nepoches', '--nepoches', help='the epoches to train.', type = int, default = 400)
parser.add_argument('-nsteps_per_epoch', '--nsteps_per_epoch', help='the steps of each epoch.', type = int, default = 100)
parser.add_argument('-lr', '--lr', help='the learning rate.', type = float, default = 0.001)
parser.add_argument('-lr_step', '--lr_step', help='the step to reduce learning rate.', type = int, default = 100)
parser.add_argument('-batch_size', '--batch_size', help='the mini-batch size.', type = int, default = 64)
parser.add_argument('-cuda_device', '--cuda_device', help='runining on specified gpu', type = int, default = 0)
parser.add_argument('-gt_treatment', '--gt_treatment', help='using treatment or not.', type = int, default = 0)#1:yes 0:no
parser.add_argument('-gt_ctimages', '--gt_ctimages', help='using ctimages or not.', type = int, default = 1)#1:yes 0:no

def build_idx_for_rankloss(bt_treatment_days, censor_indicator):
    '''
    get the index for computing rank loss. We use the non-tensorflow code for simplicity
    '''
    N = bt_treatment_days.shape[0]
    idx1 = []
    idx2 = []
    for i in range(N):
        if censor_indicator[i] == 2:
            continue
        for j in range(i+1, N):
            if bt_treatment_days[j]>bt_treatment_days[i]:
                idx1.append([i,bt_treatment_days[i]])
                idx2.append([j,bt_treatment_days[i]])
                
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    return idx1, idx2



    
@tf.function
def get_risk_reg_loss(risk_preds, gt_hitdays, event_indicator, rank_idx1, rank_idx2, total_days=62):
    '''
    the total loss for training the network: L_cured + L_dead + L_censor + L_rank
    '''
    gt_hitdays = ops.convert_to_tensor(gt_hitdays, dtype=tf.int32)
    event_indicator = ops.convert_to_tensor(event_indicator, dtype=tf.int32)
    
    H, W = risk_preds.get_shape().as_list()             
    one_hot = tf.one_hot(gt_hitdays, total_days)
    omega = 1e-5
    
    #loss for cured samples  
    keep_cured = tf.where(event_indicator==1)  
   
    cured_preds = tf.gather(risk_preds, keep_cured)
    cured_labels = tf.gather(one_hot, keep_cured)  
    cured_loss = -1.0*tf.reduce_sum(cured_labels*tf.math.log(cured_preds+omega))/tf.reduce_sum(cured_labels)
    
    #cumulative incidence function
    triu_mask = ops.convert_to_tensor(np.triu(np.ones((W,W), dtype=np.float32),0))                 
    CIF =tf.matmul(risk_preds, triu_mask)
   
    #loss for dead samples
    keep_dead = tf.where(event_indicator==2)
    dead_CIF = tf.gather(CIF, keep_dead)    
    dead_labels = np.zeros((H,W),dtype=np.float32)
    dead_labels[:,-1] = 1
    dead_labels = ops.convert_to_tensor(dead_labels)
    dead_labels = tf.gather(dead_labels, keep_dead)
    dead_loss = -1.0*tf.reduce_sum((1.0-dead_labels)*tf.math.log(1.0-dead_CIF+omega))/tf.reduce_sum(1.0-dead_labels)   
    dead_loss = tf.where(tf.math.is_nan(dead_loss),0.0, dead_loss)
    
    #loss for censored samples
    keep_censored = tf.where(event_indicator==0)
    censored_CIF = tf.gather(CIF, keep_censored)    
    censored_labels = tf.gather(one_hot, keep_censored)
    censored_loss = -1.0*tf.reduce_sum(censored_labels*tf.math.log((1.0-censored_CIF)+omega))/tf.reduce_sum(censored_labels)
    censored_loss = tf.where(tf.math.is_nan(censored_loss), 0.0, censored_loss)
    
    #rank loss
    CIF_i = tf.gather_nd(CIF, ops.convert_to_tensor(rank_idx1))
    CIF_j = tf.gather_nd(CIF, ops.convert_to_tensor(rank_idx2))    
    deta = 0.2
    CIF_d = (CIF_j-CIF_i)/deta
    rank_loss = math_ops.cast(tf.reduce_mean(tf.exp(CIF_d)), risk_preds.dtype)   
    
    return cured_loss, censored_loss, dead_loss, rank_loss


def print_estimation(bt_treatment_days, bt_estimations, bt_indicator):
    '''
    Printing information for online evaluation
    '''
    inds = np.random.choice(bt_estimations.shape[0], size=20)
    events = {0:'censor', 1:'cured', 2:'dead'}    
    for ind in inds:
        prob_within7 = np.sum(bt_estimations[ind,0:7])
        prob_within14 = np.sum(bt_estimations[ind,0:14])
        prob_within21 = np.sum(bt_estimations[ind,0:21])
        prob_within28 = np.sum(bt_estimations[ind,0:28])
        ttime = bt_treatment_days[ind] + 1
        indicator = bt_indicator[ind]
        pred_day = np.argmax(bt_estimations[ind,...]) + 1        
   
        print('event:'+ events[indicator] +'   gt:'+str(ttime) +' ' + 'pred:'+str(pred_day) + ' ' +'7days:'\
              +str(prob_within7*100)[0:6]+'%    14days:'+str(prob_within14*100)[0:6]\
              +'%    21days:'+str(prob_within21*100)[0:6]+'%   28days:'+str(prob_within28*100)[0:6]+'%')
    
def main(args):
    '''
    begin training the network
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    train_files = []
    train_subsets = args.train_subsets.split(';')
    for i in range(len(train_subsets)):
        train_files.append(os.path.join(cfg.data_set, 'subset_' + train_subsets[i] + '.csv'))
   
    eval_files = []
    eval_subsets = args.eval_subsets.split(';')
    for i in range(len(eval_subsets)):
        eval_files.append(os.path.join(cfg.data_set, eval_subsets[i]))

    if not os.path.exists(cfg.CHECKPOINTS_ROOT):
        os.mkdir(cfg.CHECKPOINTS_ROOT)
   
    train_data_generator = DataGenerator(train_files, cfg, train_mode=True)
    train_data_generator.load_dataset() 
    
    val_data_generator = DataGenerator(eval_files, cfg, train_mode=False)
    eval_sample_num = val_data_generator.load_dataset()
   
    treattype = {0:'WithoutTreatment', 1:'WithTreatment'}
    imagetype = {0:'Withoutimage', 1:''}
    if args.gt_ctimages==1:
        model_dir_name = args.train_subsets +'_modelweights_'+ treattype[args.gt_treatment]
    else:
        model_dir_name = args.train_subsets +'_modelweights_'+ treattype[args.gt_treatment]+'_'+imagetype[args.gt_ctimages]
   
    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, model_dir_name)
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
  
    model = build_network([cfg.treatment_infosize, cfg.im_feedsize, cfg.patient_infosize], [cfg.time_range], True, name='risk_predictor')
    
    if args.load_pretrain !=0:
        checkpoint_file = find_weights_of_last(os.path.join(cfg.CHECKPOINTS_ROOT, model_dir_name), 'risk_predictor')
        if checkpoint_file != '':  
            print ('loading pretrained from ', checkpoint_file)
            model.load_weights(checkpoint_file)
   
    print (model.summary())   
    
    #learning configrations
    lr_schedule =  keras.optimizers.schedules.ExponentialDecay(args.lr, args.lr_step, decay_rate = 0.96, staircase = True)
    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)    
   
    summary_writer = tf.summary.create_file_writer(model_dir)
    with summary_writer.as_default(): 
        min_eval_loss = 1000
        for epoch in range(args.nepoches):
            print ('# epoch:'+str(epoch+1)+'/'+str(args.nepoches))           
         
            risk_reg_cured_losses = []  
            risk_reg_dead_losses = []
            risk_reg_censored_losses = []
            risk_reg_rank_losses = []
            
            #training one epoch with pre-defined steps
            for step in range(args.nsteps_per_epoch):
                bt_patientinfo, bt_treatinfo, bt_ims, bt_treatment_days, bt_event_indicator, bt_severity, _ = train_data_generator.next_batch(args.batch_size)                
            
                if args.gt_treatment==0:
                    feed_treatinfo= tf.zeros_like(bt_treatinfo)
                else:
                    feed_treatinfo = bt_treatinfo    
                    
                if args.gt_ctimages==0:
                    feed_ims = tf.zeros_like(bt_ims)
                else:
                    feed_ims = bt_ims
               
                feed = [feed_treatinfo, feed_ims, bt_patientinfo]
             
                with tf.GradientTape(persistent=False) as tape:
                    risk_reg_preds = model(feed, training=True)                    
                   
                    rank_idx1, rank_idx2 = build_idx_for_rankloss(bt_treatment_days, bt_event_indicator)
                    cured_loss, censored_loss, dead_loss, rank_loss = \
                                            get_risk_reg_loss(risk_reg_preds, bt_treatment_days, bt_event_indicator, rank_idx1, rank_idx2, cfg.time_range)
                    risk_reg_cured_losses.append(cured_loss)
                    risk_reg_dead_losses.append(dead_loss)
                    risk_reg_censored_losses.append(censored_loss)
                    risk_reg_rank_losses.append(rank_loss)
                    
                    total_loss = 2*cured_loss + censored_loss + 5*dead_loss + rank_loss
                    
                    grad = tape.gradient(total_loss, model.trainable_variables)                   
                    optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))                    
          
                draw_progress_bar(step+1, args.nsteps_per_epoch)
            print('\n')                
          
           
            mean_risk_reg_cured_loss = tf.reduce_mean(risk_reg_cured_losses)
            mean_risk_reg_dead_loss = tf.reduce_mean(risk_reg_dead_losses)
            mean_risk_reg_censored_loss = tf.reduce_mean(risk_reg_censored_losses)
            mean_risk_reg_rank_loss = tf.reduce_mean(risk_reg_rank_losses)
         
            #start online evalution and save weights          
            eval_risk_reg_preds = []
            eval_treatment_days = []
            eval_event_indicators = []
            eval_gt_severitys = []
            for step in range(eval_sample_num//(args.batch_size)):
                evbt_painfo, evbt_treatinfo, evbt_ims, evbt_treatment_days, evbt_event_indicator, evbt_severity, _ = val_data_generator.next_batch(args.batch_size)
                if args.gt_treatment==0:
                    feed_treatinfo= tf.zeros_like(evbt_treatinfo)
                else:
                    feed_treatinfo = evbt_treatinfo   
                    
                if args.gt_ctimages==0:
                    feed_ims = tf.zeros_like(evbt_ims)
                else:
                    feed_ims = evbt_ims
               
                feed = [feed_treatinfo, feed_ims, evbt_painfo]
                
                risk_preds = model(feed, training=False)                
             
                eval_risk_reg_preds.append(risk_preds)
                eval_treatment_days.append(evbt_treatment_days)
                eval_event_indicators.append(evbt_event_indicator)
                eval_gt_severitys.append(evbt_severity)
               
       
            eval_risk_reg_preds =  np.concatenate(eval_risk_reg_preds, axis=0) 
            eval_treatment_days =  np.concatenate(eval_treatment_days, axis=0) 
           
            eval_event_indicators =  np.concatenate(eval_event_indicators, axis=0) 
            eval_gt_severitys =  np.concatenate(eval_gt_severitys, axis=0)
            
            print_estimation(eval_treatment_days, eval_risk_reg_preds, eval_event_indicators)
         
           
            rank_idx1, rank_idx2 = build_idx_for_rankloss(eval_treatment_days, eval_event_indicators)
            eval_cured_loss, eval_censored_loss, eval_dead_loss, eval_rank_loss = \
                                        get_risk_reg_loss(eval_risk_reg_preds, eval_treatment_days, eval_event_indicators, rank_idx1, rank_idx2, cfg.time_range)
            
           
                       
            eval_pred_days = np.argmax(eval_risk_reg_preds, axis=-1)
             
            mean_day_distance = np.mean(np.abs(eval_pred_days-eval_treatment_days))
            print ('Lr: %f \n'\
                   '|TRAN: Cured_Reg (%f); Dead_Reg (%f); Censored_Reg (%f); Rank_Reg (%f) \n'\
                   '|EVAL: Cured_Reg (%f); Dead_Reg (%f); Censored_Reg (%f); Rank_Reg (%f); \n'\
                   '       MD-DIS (%f) \n' % \
                   ( KB.eval(optimizer._decayed_lr('float32')),
                     mean_risk_reg_cured_loss, mean_risk_reg_dead_loss, mean_risk_reg_censored_loss, mean_risk_reg_rank_loss, \
                     eval_cured_loss, eval_dead_loss, eval_censored_loss, eval_rank_loss, mean_day_distance)) 
            
            
            
            #eval_total_loss = eval_severity_cls_loss + eval_cured_loss + eval_censored_loss + eval_dead_loss + eval_rank_loss
            if  mean_day_distance<=min_eval_loss:             
                min_eval_loss = mean_day_distance
                print('saving model weights to checkpoint!', model_dir)
                checkpoint_path = osp.join(model_dir,  "risk_predictor_epoch{0}.h5".format(epoch)) 
                model.save_weights(checkpoint_path)
                delete_old_weights(model_dir, 'risk_predictor', cfg.MAX_KEEPS_CHECKPOINTS)

    
if __name__ == "__main__":
    args = parser.parse_args()
    print (args)
    main(args)
