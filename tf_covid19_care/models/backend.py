# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:40:06 2020

@author: wjcongyu
"""
import glob
import os
import os.path as osp
import sys

def find_weights_of_last(model_dir, net_name=''):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        the path of the last checkpoint file
    """
    weights_files = glob.glob(osp.join(model_dir, net_name.lower() + '*.h5'))
    if len(weights_files) == 0:
        return ''
    weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))
    return weights_files[-1]

def delete_old_weights(model_dir, net_name='', nun_max_keep=1):
    weights_files = glob.glob(osp.join(model_dir, net_name.lower() + '*.h5'))
    if len(weights_files) <= nun_max_keep:
        return

    weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))

    weights_files = weights_files[0:len(weights_files) - nun_max_keep]

    for weight_file in weights_files:
        os.remove(weight_file)
        
        
def draw_progress_bar(cur, total, bar_len=50):
    cur_len = int(cur/total*bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()