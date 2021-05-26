# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:24:43 2019
The class of unet_3d for the centernet detection
@author: wjcongyu
"""
import os
import os.path as osp
import tensorflow as tf
from tensorflow import math
from tensorflow import keras
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import losses as KLOSS
from tensorflow.keras import backend as KB
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import losses_utils
import glob
import numpy as np
import sys

class Seg_3DUnet():
    def __init__(self,input_shape, is_training, config, num_classes, model_dir):
        self._is_training = is_training
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.NET_NAME = 'Seg_3DUnet'
        self.__set_log_dir() #logging and saving checkpoints
        self.model = self.__build(is_training=is_training)

    #public functions
    def summary(self):
        '''
        print the network attributes
        :return:
        '''
        return self.model.summary()

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        weights_files = glob.glob(osp.join(self.log_dir, self.NET_NAME.lower() + '*.h5'))
        if len(weights_files) == 0:
            return ''
        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))
        return weights_files[-1]

    def load_weights(self, filepath, by_name=False, exclude=None):
        '''
        loading weights from checkpoint
        :param filepath:
        :param by_name:
        :param exclude:
        :return:
        '''
        print('loading weights from:', filepath)
        self.model.load_weights(filepath, by_name)

    def train(self,  train_data_provider, val_data_provider, learning_rate, decay_steps, epochs, batch_size,
              augment=None, custom_callbacks=None):
        '''
        Start training the model from specified dataset
        :param train_dataset:
        :param learning_rate:
        :param decay_steps:
        :param epochs:
        :param augment:
        :param custom_callbacks:
        :return:
        '''
        assert self._is_training == True, 'not in training mode'


        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)

        lr_schedule =  keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                   decay_steps,
                                                                   decay_rate = 0.95,
                                                                   staircase = True)
        optimizer =keras.optimizers.Adam(learning_rate = lr_schedule)
        #optimizer = keras.optimizers.SGD(lr=learning_rate, decay= learning_rate/decay_steps, momentum=0.92, nesterov=True)# 
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        with self.summary_writer.as_default(): 
            max_accuracy = 0.0
            for self.epoch in range(epochs):
                print ('# epoch:'+str(self.epoch+1)+'/'+str(epochs))
                losses = []
                for step in range(self.config.STEPS_PER_EPOCH):
                    ims, label_gt = train_data_provider.next_batch(batch_size)
                    with tf.GradientTape(persistent=False) as tape:
                        label_preds = self.model(ims)
                        loss = self.__compute_loss(label_gt, label_preds)
                        
                        losses.append(loss)
                        grad = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.trainable_variables))
                        self.__draw_progress_bar(step+1,self.config.STEPS_PER_EPOCH)
                    
                tst_ims, tst_y_gt = val_data_provider.next_batch(1)
               
                predictions = self.predict(tst_ims)
                
                label_preds = tf.reshape(predictions,[-1, self.num_classes])
                gt_labels = tf.reshape(tst_y_gt, [-1])
        
                pred_labels = tf.argmax(label_preds, axis=-1)
                
                keep = tf.where(tf.not_equal(gt_labels, 0))
                gt_labels = tf.gather(gt_labels, keep)        
           
                pred_labels = tf.gather(pred_labels, keep)
                
                m = keras.metrics.Accuracy()
                m.update_state(gt_labels, pred_labels)
                accuracy = m.result().numpy()
                mean_loss = tf.reduce_mean(losses)
                print ('\nLoss:%f; Accuracy:%f; Lr: %f' % (mean_loss, accuracy, KB.eval(optimizer._decayed_lr('float32'))))
                tf.summary.scalar('train_loss', mean_loss, step = (self.epoch+1))
                tf.summary.scalar('eval_accuracy', float(accuracy), step = (self.epoch+1))
                    
                m.reset_states()
                if accuracy >= max_accuracy or accuracy > 0.98:
                    max_accuracy = accuracy
                    self.checkpoint_path = osp.join(self.log_dir, self.NET_NAME.lower() + "_epoch{0}.h5".format(self.epoch + 1))            
                    print ('Saving weights to %s' % (self.checkpoint_path))
                    self.model.save_weights(self.checkpoint_path)
                    self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

             
                '''mean_loss = tf.reduce_mean(losses)
                print ('\nLoss:%f; Lr: %f' % (mean_loss, KB.eval(optimizer._decayed_lr('float32'))))
                tf.summary.scalar('train_loss', mean_loss, step = (self.epoch+1))              
                   
                self.checkpoint_path = osp.join(self.log_dir, self.NET_NAME.lower() + "_epoch{0}.h5".format(self.epoch + 1))            
                print ('Saving weights to %s' % (self.checkpoint_path))
                self.model.save_weights(self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)'''

    def predict(self, image):
        return self.model.predict(image)

    #private functions
    def __set_log_dir(self):
        self.epoch = 0
        self.log_dir = osp.join(self.model_dir, self.NET_NAME.lower())

    def __build(self, is_training):
        #define inputs:[batch_size, Depth, Height, Width, Channels], for keras, you don't need
        #to specify the batch_size
        dtype = tf.float32
        input_image = KL.Input(shape = self.input_shape + [1], dtype= dtype, name='input_image')
        filters = [16, 32, 64, 128]
        x1 = KL.Conv3D(filters[0]//2, (3, 3, 3), (1, 1, 1), padding='same')(input_image)   
        x1 = KL.BatchNormalization(axis=-1)(x1, training=is_training)
        x1 = KL.ReLU()(x1)        
        x1 = KL.Conv3D(filters[0], (3, 3, 3), (1, 1, 1), padding='same')(x1)   
        x1 = KL.BatchNormalization(axis=-1)(x1, training=is_training)
        x1 = KL.ReLU()(x1)
        d1 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x1)       
        
        x2 = KL.Conv3D(filters[1]//2, (3, 3, 3), (1, 1, 1), padding='same')(d1)   
        x2 = KL.BatchNormalization(axis=-1)(x2, training=is_training)
        x2 = KL.ReLU()(x2)        
        x2 = KL.Conv3D(filters[1], (3, 3, 3), (1, 1, 1), padding='same')(x2)   
        x2 = KL.BatchNormalization(axis=-1)(x2, training=is_training)
        x2 = KL.ReLU()(x2)
        d2 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x2)
       
        x3 = KL.Conv3D(filters[2]//2, (3, 3, 3), (1, 1, 1), padding='same')(d2)   
        x3 = KL.BatchNormalization(axis=-1)(x3, training=is_training)
        x3 = KL.ReLU()(x3)        
        x3 = KL.Conv3D(filters[2], (3, 3, 3), (1, 1, 1), padding='same')(x3)   
        x3 = KL.BatchNormalization(axis=-1)(x3, training=is_training)
        x3 = KL.ReLU()(x3)
        d3 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x3)
        
        x4 = KL.Conv3D(filters[3]//2, (3, 3, 3), (1, 1, 1), padding='same')(d3)   
        x4 = KL.BatchNormalization(axis=-1)(x4, training=is_training)
        x4 = KL.ReLU()(x4)        
        x4 = KL.Conv3D(filters[3], (3, 3, 3), (1, 1, 1), padding='same')(x4)   
        x4 = KL.BatchNormalization(axis=-1)(x4, training=is_training)
        x4 = KL.ReLU()(x4)
        d4 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x4)
       
        u5 = KL.UpSampling3D()(d4)
        x5 = KL.Conv3D(filters[3]//2, (3, 3, 3), (1, 1, 1), padding='same')(u5)   
        x5 = KL.BatchNormalization(axis=-1)(x5, training=is_training)
        x5 = KL.ReLU()(x5)
        x5 = KL.Conv3D(filters[3]//2, (3, 3, 3), (1, 1, 1), padding='same')(x5)   
        x5 = KL.BatchNormalization(axis=-1)(x5, training=is_training)
        x5 = KL.ReLU()(x5)
        m5 = KL.Concatenate()([x5,x4])
        x5 = KL.Conv3D(filters[3], (3, 3, 3), (1, 1, 1), padding='same')(m5)   
        x5 = KL.BatchNormalization(axis=-1)(x5, training=is_training)
        x5 = KL.ReLU()(x5)
        
        u6 = KL.UpSampling3D()(x5)
        x6 = KL.Conv3D(filters[2]//2, (3, 3, 3), (1, 1, 1), padding='same')(u6)   
        x6 = KL.BatchNormalization(axis=-1)(x6, training=is_training)
        x6 = KL.ReLU()(x6)
        x6 = KL.Conv3D(filters[2]//2, (3, 3, 3), (1, 1, 1), padding='same')(x6)   
        x6 = KL.BatchNormalization(axis=-1)(x6, training=is_training)
        x6 = KL.ReLU()(x6)
        m6 = KL.Concatenate()([x6,x3])
        x6 = KL.Conv3D(filters[2], (3, 3, 3), (1, 1, 1), padding='same')(m6)   
        x6 = KL.BatchNormalization(axis=-1)(x6, training=is_training)
        x6 = KL.ReLU()(x6)
        
        u7 = KL.UpSampling3D()(x6)
        x7 = KL.Conv3D(filters[1]//2, (3, 3, 3), (1, 1, 1), padding='same')(u7)   
        x7 = KL.BatchNormalization(axis=-1)(x7, training=is_training)
        x7 = KL.ReLU()(x7)
        x7 = KL.Conv3D(filters[1]//2, (3, 3, 3), (1, 1, 1), padding='same')(x7)   
        x7 = KL.BatchNormalization(axis=-1)(x7, training=is_training)
        x7 = KL.ReLU()(x7)
        m7 = KL.Concatenate()([x7,x2])
        x7 = KL.Conv3D(filters[1], (3, 3, 3), (1, 1, 1), padding='same')(m7)   
        x7 = KL.BatchNormalization(axis=-1)(x7, training=is_training)
        x7 = KL.ReLU()(x7)
        
        u8 = KL.UpSampling3D()(x7)
        x8 = KL.Conv3D(filters[0]//2, (3, 3, 3), (1, 1, 1), padding='same')(u8)   
        x8 = KL.BatchNormalization(axis=-1)(x8, training=is_training)
        x8 = KL.ReLU()(x8)
        x8 = KL.Conv3D(filters[0]//2, (3, 3, 3), (1, 1, 1), padding='same')(x8)   
        x8 = KL.BatchNormalization(axis=-1)(x8, training=is_training)
        x8 = KL.ReLU()(x8)
        m8 = KL.Concatenate()([x8,x1])
        x8 = KL.Conv3D(filters[0], (3, 3, 3), (1, 1, 1), padding='same')(m8)   
        x8 = KL.BatchNormalization(axis=-1)(x8, training=is_training)
        x8 = KL.ReLU()(x8)
        
        #define output logits
        x9 = KL.Conv3D(self.num_classes, [3, 3, 3], padding='same')(x8)   
        output = KL.Reshape([-1, self.num_classes])(x9)
        output = KL.Activation('softmax')(output)
        output = KL.Reshape(self.config.INPUT_SHAPE + [self.num_classes])(output)
        
        model = KM.Model(input_image, output, name=self.NET_NAME.lower())
        return model

   
    def __compute_loss(self, label_gt, label_preds):
        '''
        the loss for center keypoint loss
        :param cnt_gt:
        :param cnt_preds:
        :return:
        '''
        label_preds = ops.convert_to_tensor(label_preds)
        label_gt = math_ops.cast(label_gt, label_preds.dtype)
        label_preds = tf.reshape(label_preds,[-1, self.num_classes])
        label_gt = tf.reshape(label_gt, [-1, 1])
       
        return KLOSS.SparseCategoricalCrossentropy()(label_gt, label_preds)


    def __delete_old_weights(self, nun_max_keep):
        '''
        keep num_max_keep weight files, the olds are deleted
        :param nun_max_keep:
        :return:
        '''
        weights_files = glob.glob(osp.join(self.log_dir, self.NET_NAME.lower() + '*.h5'))
        if len(weights_files) <= nun_max_keep:
            return

        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))

        weights_files = weights_files[0:len(weights_files) - nun_max_keep]

        for weight_file in weights_files:
            if weight_file != self.checkpoint_path:
                os.remove(weight_file)

    def __draw_progress_bar(self, cur, total, bar_len=50):
        cur_len = int(cur/total*bar_len)
        sys.stdout.write('\r')
        sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
        sys.stdout.flush()