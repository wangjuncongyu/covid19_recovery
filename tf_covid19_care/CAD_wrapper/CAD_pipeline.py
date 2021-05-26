import _init_pathes
import logging
import datetime
import os.path as osp
import numpy as np
from configs.lung_cfgs import cfg as lung_cfg
from .lung_segmentor import LungSegmentor
from configs.cfgs import cfg as risk_cfg
from .risk_predictor import RiskPredictor
from data_process.data_processor import get_series_uids
from data_process.data_processor import load_series_volume
from data_process.data_processor import interpolate_volume
from data_process.data_processor import get_series_dcm_nums
from data_process.data_processor import get_convex_bbox_frm_3dmask
from data_process.data_processor import resize, hu2gray
import matplotlib.pyplot as plt

class CAD_Pipeline(object):
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='nodule_detector.log',
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def initialize(self, debug_mode = True):
        '''
        Before pulmonary lesion detection, you have to call this function to initialize the networks.
        '''
        self.debug_mode = debug_mode
        self.is_ready = False
        #init lung segementation network
        self.lung_segmentor = LungSegmentor(lung_cfg)
        if not self.lung_segmentor.initialize():
            self._record_error_info(self.lung_segmentor.last_error)
            self.is_ready = False
            return False
        
        
        self.risk_predictor = RiskPredictor(risk_cfg)
        if not self.risk_predictor.initialize():
            self._record_error_info(self.risk_predictor.last_error)
            self.is_ready = False
            return False
        
        if osp.exists('../checkpoints/feature_minv.npy'):
            print('found minv file for normalization')
            self.patient_infominv = np.load('../checkpoints/feature_minv.npy', allow_pickle=True)
            
        else:
            self.patient_infominv = np.min(self.patients[:, 1:48], axis=0)
        self.patient_infominv[1] = 1
        
        if osp.exists('../checkpoints/feature_maxv.npy'):
            print('found maxv file for normalization')
            self.patient_infomaxv = np.load('../checkpoints/feature_maxv.npy', allow_pickle=True)
           
        else:
            self.patient_infomaxv = np.max(self.patients[:, 1:48], axis=0)
        self.patient_infomaxv[1] = 100
        
        self.patient_infominv = np.delete(self.patient_infominv, 2, axis=0)
        self.patient_infomaxv = np.delete(self.patient_infomaxv, 2, axis=0)
        self.is_ready = True
        self._record_debug_info('Networks are ready!')
        return self.is_ready
    
    

    def predict_risk_for(self, patient_root):
        '''
        Detection on scan of dcm files
        '''
        assert self.is_ready==True, 'Networks not ready!'
        if not osp.exists(patient_root):
            self._record_error_info('No patient path found at ' + patient_root)
            return False
        treatment_file = osp.join(patient_root, 'treatments.npy')
        clinical_file = osp.join(patient_root, 'clinical_features.npy')
        
        if not osp.exists(treatment_file) or not osp.exists(clinical_file):
            self._record_error_info('treatments.npy and clinical_features.npy required, but no file was found at ' + patient_root)
            return False
        try:  
            treatments = np.load(treatment_file, allow_pickle=True).astype(np.float32) 
            treatments = treatments.reshape((1, -1))
            clinicals = np.load(clinical_file, allow_pickle=True)
            int_clinicals = np.int32(clinicals.copy())
            keep = np.where(int_clinicals>0)
         
            clinicals = (clinicals-self.patient_infominv)/(self.patient_infomaxv-self.patient_infominv)
            clinicals[clinicals>1.0] = 1.0
            clinicals[clinicals<0.0] = 0.0
            clinicals = clinicals.reshape(1, -1)
            #processing ct scan if it exists
            feed_volume = self._try_to_load_ct(osp.join(patient_root, 'ct'))
            if feed_volume is None:
                feed_volume = np.zeros([1]+risk_cfg.im_feedsize+ [1], dtype=np.float32)
         
            
            feature_significane_coff, reg_pred = self.risk_predictor.predict(treatments,feed_volume, clinicals)
            feature_significane_coff = feature_significane_coff[0,...]
           
            reg_pred = reg_pred[0,...]
            if feature_significane_coff is None:
                return False
            
            np.save(osp.join(patient_root, 'predictions.npy'), reg_pred)
    
            self._generate_figures(patient_root, keep, feature_significane_coff, reg_pred)
            return True
        except Exception as e:
            self._record_error_info(str(e))
                        
            return False
            
       
    def _try_to_load_ct(self, ct_root):
        if not osp.exists(ct_root):
            return None
        try:
            series_uids = get_series_uids(ct_root)
            if series_uids is None or len(series_uids)==0:
                self._record_error_info('No CT scan found at ' + ct_root) 
                return None  
                     
            feed_volume = None
            for series_uid in series_uids:
                n_dcmfiles = get_series_dcm_nums(ct_root, series_uid)
                if n_dcmfiles < 60:
                    self._record_debug_info('CT scan ignored due to the number of dcm files<50 of series:' + series_uid)
                    continue
                   
                self._record_debug_info('Start loading series:' + series_uid)
                                              
                volume, origin, spacing, slice_spacing, im_numbers = load_series_volume(ct_root, series_uid)
                if volume is None:
                    self._record_error_info('Load volume data failed.')                                
                    return None
                new_spacing = [0.65, 0.65, 1.5]
                new_volume = interpolate_volume(volume.copy(), spacing, new_spacing)
                nD,nH,nW = new_volume.shape
         
                self._record_debug_info('Lung segmentation ...')
                lung_mask_vol = self.lung_segmentor.segment_lung(new_volume)
                if lung_mask_vol is None or np.count_nonzero(lung_mask_vol)<(nD*nH*nW*0.01):
                    return None
                                
                bbox = get_convex_bbox_frm_3dmask(lung_mask_vol, [2,2,2])
                
                feed_volume = resize(hu2gray(new_volume[bbox[2]:bbox[5], bbox[1]:bbox[4], bbox[0]:bbox[3]],WL=-500, WW=1200), risk_cfg.im_feedsize)
                feed_volume = feed_volume.reshape((1,)+feed_volume.shape + (1,))
                break
            return feed_volume
        except Exception as e:
            self._record_error_info(str(e))                        
            return None
    
    def _generate_figures(self, save_path, keep, coff, reg_pred):
        font={'family':'Arial',
              'style':'normal',
              'weight':'light',
              'color':'black',
              'size':13
              }
        
        plt.figure()
        #colors = ['#7281a7', '#af97ba', '#ef626c', '#ef626c']
        if reg_pred[-1]>0.5:
            color = 'red'
        else:
            color = 'green'
            
        pred_hitday = np.argmax(reg_pred)+1
        plt.plot([x for x in range(1,33)], reg_pred, color=color, label='Predicted probability distribution', linestyle='-', linewidth=3)
        plt.plot([pred_hitday, pred_hitday], [0, reg_pred[pred_hitday-1]], 'k--', lw=1, label='Predicted recovery day: ' + str(pred_hitday))
  
        plt.xlim([0, 33])
        plt.ylim([0, 0.5])
        plt.xlabel('Days', fontdict=font)
        plt.ylabel('Recovery probability', fontdict=font)
        plt.title('', fontdict=font)
   
        plt.legend(loc='upper right', prop={'size':14, 'family':'arial'})

        plt.savefig(osp.join(save_path, 'recovery_probability.png'), dpi=300)
        
        #saving top-5 features
        
        plt.figure()
        
        feature_names = np.array(['Gender', 'Age', 'Fever', 'Cough', 'Soreness', 'Weakness',\
                 'Headache', 'Nausea/Vomiting', 'Diarrhea', 'Expectoration',\
                 'Chest Congestion', 'Poor Appetite' ,'Poor Spirits',\
                 'Diabetes', 'Kidney Disease', 'Malignant Tumor',
                 'Chronic Hepatitis B' ,'Hypothyroidism', 'ARDS',
                 'SK' ,'IL-6' ,'WBC' ,'LAV',
                 'NAV', 'HG' ,'RBC',
                 'PC' ,'HS-CRP', 'CRP',
                 'PCT' ,'PT', 'TT',
                 'APTT' ,'FB' ,'DD', 'BGLU',
                 'ALT' ,'AST', 'TP',
                 'AM' ,'TB' ,'DB', 'UN',
                 'CRT', 'LDH', 'CK',
                 'CK-MB' ,'α-HBDH', 'CT'])
        
        colors = ['darkviolet', 'darkorange','aqua', 'yellowgreen']
        color_idx =[1, 1, 2, 2, 2, 2,\
                    2, 2, 2, 2,\
                    2, 2 ,2,\
                    0, 0, 0,
                    0 ,0, 0,
                    0 ,3 ,3 ,3,
                    3, 3 ,3,
                    3 ,3, 3,
                    3 ,3, 3,
                    3 ,3 ,3, 3,
                    3 ,3, 3,
                    3 ,3 ,3, 3,
                    3, 3, 3,
                    3 ,3,
                    3]
        name_colors = []
        for i in color_idx:
            name_colors.append(colors[i])
        name_colors = np.array(name_colors)
        
        coff = coff[keep]
        feature_names = feature_names[keep]
        name_colors = name_colors[keep]
       
        top10_ind = np.argsort(-np.array(coff))[0:8]
        
        top10_coffe = coff[top10_ind]
       
        top10_names = np.array(feature_names)[top10_ind] 
        top10_colors = name_colors[top10_ind]
        
        plt.rcdefaults()
        fig, ax = plt.subplots()
    
        y_pos = np.arange(len(top10_names))
        ax.barh(y_pos, top10_coffe, align='center', color=top10_colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_xlim(xmax=1.0, xmin=0)
        ax.set_yticklabels(top10_names, font)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Average impact on model output', font)
   
        plt.tight_layout()
        plt.savefig(osp.join(save_path, 'top5.png'), dpi=300)
        
        
   
    def _record_error_info(self, error_info):
        logging.error(error_info)
        if self.debug_mode:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print (error_info)

    def _record_debug_info(self, debug_info):
        if self.debug_mode:
            logging.info(debug_info)
            print (debug_info)
