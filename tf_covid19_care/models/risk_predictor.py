# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:43:08 2020

@author: wjcongyu
"""

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM

def SA_Module(inputs, training, name ='SA_Module'):
    '''
    the self-attentino module
    '''
    hidden1 = KL.Dense(512, activation='relu', name=name+'hidden1')(inputs)
   
    t_v = KL.Dense(inputs.get_shape().as_list()[-1] , activation='selu',  use_bias=False,\
                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=1.0, maxval=1.0),\
                   kernel_regularizer = 'l1', name=name+'_t_v')(hidden1)      #, #kernel_regularizer = 'l1', \
    t_softmax = KL.Softmax(name=name+'_t_softmax')(t_v)  
    t_inputs = inputs * t_softmax
    return t_inputs



def build_network(input_shapes, output_size, training, name = 'TreatmentRecommder'):
    '''
    build the network for covid-19 prediction of how long a patient can be cured
    '''
    dtype = tf.float32
    #treatment information
    treatment_info = KL.Input(shape = input_shapes[0], dtype = dtype, name='treatment_info') 
   
    #imaing information: CNN features from CT images
    image_info = KL.Input(shape = input_shapes[1]+[1], dtype = dtype, name='image_info')   
    base_filters = 8    
    x11 = KL.Conv3D(base_filters, (3, 3, 3), padding='same', name = 'x11')(image_info)  
    x11 = KL.BatchNormalization(axis=-1, name = 'bn11')(x11, training=training)
    x11 = KL.ReLU(name = 'relu11')(x11)
    x12 = KL.Conv3D(base_filters, (3, 3, 3), padding='same', name = 'x12')(x11)  
    x12 = KL.BatchNormalization(axis=-1, name = 'bn12')(x12, training=training)
    x12 = KL.ReLU(name = 'relu12')(x12)
    d1 = KL.MaxPool3D()(x12)
    
    x21 = KL.Conv3D(base_filters*2, (3, 3, 3), padding='same', name = 'x21')(d1) 
    x21 = KL.BatchNormalization(axis=-1, name = 'bn21')(x21, training=training)
    x21 = KL.ReLU(name = 'relu21')(x21)
    x22 = KL.Conv3D(base_filters*2, (3, 3, 3), padding='same', name = 'x22')(x21) 
    x22 = KL.BatchNormalization(axis=-1, name = 'bn22')(x22, training=training)
    x22 = KL.ReLU(name = 'relu22')(x22)
    
    d2 = KL.MaxPool3D()(x22)
    
    x31 = KL.Conv3D(base_filters*4, (3, 3, 3), padding='same', name = 'x31')(d2)  
    x31 = KL.BatchNormalization(axis=-1, name = 'bn31')(x31, training=training)
    x31 = KL.ReLU(name = 'relu31')(x31)
    x32 = KL.Conv3D(base_filters*4, (3, 3, 3), padding='same', name = 'x32')(x31)  
    x32 = KL.BatchNormalization(axis=-1, name = 'bn32')(x32, training=training)
    x32 = KL.ReLU(name = 'relu32')(x32)
    x33 = KL.Conv3D(base_filters*4, (3, 3, 3), padding='same', name = 'x33')(x32)  
    x33 = KL.BatchNormalization(axis=-1, name = 'bn33')(x33, training=training)
    x33 = KL.ReLU(name = 'relu33')(x33)
   
    d3 = KL.MaxPool3D()(x33)
    
    x41 = KL.Conv3D(base_filters*8, (3, 3, 3), padding='same', name = 'x41')(d3)  
    x41 = KL.BatchNormalization(axis=-1, name = 'bn41')(x41, training=training)
    x41 = KL.ReLU(name = 'relu41')(x41)      
    x42 = KL.Conv3D(base_filters*8, (3, 3, 3), padding='same', name = 'x42')(x41)  
    x42 = KL.BatchNormalization(axis=-1, name = 'bn42')(x42, training=training)
    x42 = KL.ReLU(name = 'relu42')(x42)
    x43 = KL.Conv3D(base_filters*8, (3, 3, 3), padding='same', name = 'x43')(x42)  
    x43 = KL.BatchNormalization(axis=-1, name = 'bn43')(x43, training=training)
    x43 = KL.ReLU(name = 'relu43')(x43)
    
    d4 = KL.MaxPool3D()(x43)
    
    x51 = KL.Conv3D(base_filters*16, (3, 3, 3), padding='same', name = 'x51')(d4)  
    x51 = KL.BatchNormalization(axis=-1, name = 'bn51')(x51, training=training)
    x51 = KL.ReLU(name = 'relu51')(x51)      
    x52 = KL.Conv3D(base_filters*16, (3, 3, 3), padding='same', name = 'x52')(x51)  
    x52 = KL.BatchNormalization(axis=-1, name = 'bn52')(x52, training=training)
    x52 = KL.ReLU(name = 'relu52')(x52)
    x53 = KL.Conv3D(base_filters*16, (3, 3, 3), padding='same', name = 'x53')(x52)  
    x53 = KL.BatchNormalization(axis=-1, name = 'bn53')(x53, training=training)
    x53 = KL.ReLU(name = 'relu53')(x53)
 
    #d5 = KL.MaxPool3D()(x52)
    cnn_GAP = KL.GlobalAveragePooling3D(name='CNN_GAP')(x53)
    #cnn_cof = KL.Dense(1, activation='relu', name='cnn_cof')(cnn_GAP)
    
    #patient information
    patient_info = KL.Input(shape = input_shapes[2], dtype = dtype, name='patient_info')
    #pcnn_info = KL.Concatenate()([patient_info, cnn_cof])    
    
    #cured probability distruibution subnetwork
    #w_pcnn_info = SA_Module(pcnn_info, training)
    w_pcnn_info = SA_Module(patient_info, training)
    
    fc1 = KL.Dense(256, activation='relu', name='fc1')(KL.Concatenate()([w_pcnn_info, cnn_GAP, treatment_info])) 
    fc2 = KL.Dense(512, activation='relu', name='fc2')(fc1) 
    fc2 = KL.Dropout(0.3)(fc2, training = training)
    fc3 = KL.Dense(512, activation='relu', name='fc3')(fc2) 
    fc3 = KL.Dropout(0.3)(fc3, training = training)
    
    fc_reg = KL.Dense(256, activation='relu', name='fc_reg')(fc3)
    
    risk_reg_preds = KL.Dense(output_size[0],activation='softmax', name='risk_reg_preds')(fc_reg)
    
    model = KM.Model([treatment_info,image_info,patient_info], [risk_reg_preds], name=name)
    return model
    
