import pandas as pd
import numpy as np
import glob
import os.path as osp

class SurvivalMetric(object):
    def __init__(self, prob_preds, true_recovery_day, thres_death=0.5):
        self.prob_preds = prob_preds
        self.pred_day = np.argmax(self.prob_preds, axis = -1)
        self.true_recovery_day = true_recovery_day
        self.thres_death = thres_death
        
        self.CIF = self.compute_CIF()
        
    def compute_CIF(self):
        H, W = self.prob_preds.shape
        triu_mask = np.triu(np.ones((W,W), dtype=np.float32),0)                 
        CIF =np.matmul(self.prob_preds, triu_mask)
        return CIF
    
    def compute_DTCI(self):
        N = self.prob_preds.shape[0]
        A = 0
        F = 0
        for i in range(N):
            for j in range(i+1, N):
                if self.true_recovery_day[i]<self.true_recovery_day[j]:
                    A+=1
                    if self.CIF[i, self.true_recovery_day[i]]>self.CIF[j, self.true_recovery_day[i]]:
                        F+=1
       
        return F/(A+0.000001)
    
    def compute_MSADIS(self):
        diff = np.abs(self.pred_day - self.true_recovery_day)
       
        meanv = np.mean(diff)
        stdv = np.std(diff)
        return meanv, stdv
    
class SurvivalMetri_bootstrap(object):
    def __init__(self, prob_preds, true_recovery_day, thres_death=0.5):
        self.prob_preds = prob_preds
        self.pred_day = np.argmax(self.prob_preds, axis = -1)
        self.true_recovery_day = true_recovery_day
        self.thres_death = thres_death
        
        self.CIF = self.compute_CIF()
        
    def compute_CIF(self):
        H, W = self.prob_preds.shape
        triu_mask = np.triu(np.ones((W,W), dtype=np.float32),0)                 
        CIF =np.matmul(self.prob_preds, triu_mask)
        return CIF
    
    def _compute_DTCI(self, sampes_pred, samples_cif, samples_true_recovery_day):
        N = sampes_pred.shape[0]
        A = 0
        F = 0
        for i in range(N):
            for j in range(i+1, N):
                if samples_true_recovery_day[i]<samples_true_recovery_day[j]:
                    A+=1
                    if samples_cif[i, samples_true_recovery_day[i]]>samples_cif[j, samples_true_recovery_day[i]]:
                        F+=1
       
        return F/(A+0.000001)
    
    def compute_DTCI(self):
        DTCIs= []
        N = self.prob_preds.shape[0]
        a = np.arange(N)
        for k in range(100):            
            samples_pred = []
            samples_true_recovery_day = []
            samples_cif = []
            for i in range(N):
                sample_idx = np.random.choice(a, 1, True)[0]
                samples_pred.append(self.prob_preds[sample_idx,...])
                samples_cif.append(self.CIF[sample_idx,...])
                samples_true_recovery_day.append(self.true_recovery_day[sample_idx])
            
            samples_pred = np.vstack(samples_pred)
           
            samples_cif = np.vstack(samples_cif)
            samples_true_recovery_day = np.array(samples_true_recovery_day)
            DTCIs.append(self._compute_DTCI(samples_pred, samples_cif, samples_true_recovery_day))
        
        DTCIs.sort()
        return np.mean(DTCIs), DTCIs[4], DTCIs[-5]
    
    def compute_MSADIS(self):
        all_diff = np.abs(self.pred_day - self.true_recovery_day)
        diffs =[]
        N = self.prob_preds.shape[0]
        a = np.arange(N)
        for k in range(100):
            samples = []
            for i in range(N):
                sample_idx = np.random.choice(a, 1, True)[0]
                samples.append(all_diff[sample_idx])
            samples = np.array(samples)
           
            diffs.append(np.mean(samples))
                
        diffs.sort()
        return np.mean(diffs), diffs[4], diffs[-5]
        
def compute_metrics(dataset = 'huoshenshan',group = 0, input_infor = 0):
    group_name = {0:'all', 1:'low-risk', 2:'high-risk', 3:'<=50 years old', 4:'>50 years old'}
    input_infoname = {0:'org', 1:'no_treat', 2:'no_im'}
    #iCOVID
    rst_root = dataset+'_rst_' + input_infoname[input_infor]
    flag = 'GT'
    if input_infor==1:
        flag='NONE'
    print('dataset|group|input:', dataset, group_name[group],input_infoname[input_infor] )
    all_prob_preds = []
    all_recovery_days = []
    DTCIs = []
    for i in range(1, 6):
        print('\n========== subset {0} =============='.format(i))
        prob_preds = np.array(pd.read_csv(glob.glob(osp.join(rst_root, '*'+str(i)+'_'+flag+'_risk_reg_preds.csv'))[0]).iloc[0:], dtype=np.float32)        
        true_recovery_days = np.array(pd.read_csv(glob.glob(osp.join(rst_root, '*'+str(i)+'_'+flag+'_gt_hitday.csv'))[0]).iloc[0:], dtype=np.int32).reshape(-1)       
        event_indicators = np.array(pd.read_csv(glob.glob(osp.join(rst_root, '*'+str(i)+'_'+flag+'_indicator.csv'))[0]).iloc[0:], dtype=np.int32).reshape(-1)
        severity = np.array(pd.read_csv(glob.glob(osp.join(rst_root, '*'+str(i)+'_'+flag+'_gt_severity.csv'))[0]).iloc[0:], dtype=np.int32).reshape(-1)
        features = np.array(pd.read_csv(glob.glob(osp.join(rst_root, '*'+str(i)+'_'+flag+'_clinic_features.csv'))[0]).iloc[0:], dtype=np.float32)
        ages = features[:,1]*119+1
        
        keep = np.where(event_indicators==1)[0]
        prob_preds = prob_preds[keep,...]
        true_recovery_days = true_recovery_days[keep]
        severity = severity[keep]
        ages = ages[keep]
        
        keep = np.where(severity!=10)[0]
        if group == 1:        
            keep = np.where(severity<=1)[0]
        elif group ==2:
            keep = np.where(severity>1)[0]
        elif group ==3:
            keep = np.where(ages<=50)[0]
        elif group ==4:
            keep = np.where(ages>50)[0]
        
            
        prob_preds = prob_preds[keep,...]
        true_recovery_days = true_recovery_days[keep]
        
        all_prob_preds.append(prob_preds)
        all_recovery_days.append(true_recovery_days)
        metric_computer = SurvivalMetri_bootstrap(prob_preds, true_recovery_days)
       
        dtci, dtci_c1, dtci_c2 = metric_computer.compute_DTCI()
       
        msadis, msadis_c1, msadis_c2 = metric_computer.compute_MSADIS()
        print('patients:', prob_preds.shape[0])
        print('DT-CI:{0} [{1}, {2}]'.format(dtci, dtci_c1, dtci_c2 ))
        print('MS-ADS:{0} [{1}, {2}]'.format(msadis, msadis_c1, msadis_c2))
        
    all_prob_preds = np.concatenate(all_prob_preds, axis=0)   
    all_recovery_days = np.concatenate(all_recovery_days, axis=0)
    
    metric_computer = SurvivalMetri_bootstrap(all_prob_preds, all_recovery_days)
       
    dtci, dtci_c1, dtci_c2 = metric_computer.compute_DTCI()
       
    msadis, msadis_c1, msadis_c2 = metric_computer.compute_MSADIS()
    print('\n=========== overall ==============')
    print('patients:',all_prob_preds.shape[0])
    print('DT-CI:{0} [{1}, {2}]'.format(dtci, dtci_c1, dtci_c2 ))
    print('MS-ADS:{0} [{1}, {2}]'.format(msadis, msadis_c1, msadis_c2))
   
   
    
def main():
    compute_metrics('huoshenshan',0, 2)
    
if __name__ == '__main__':
    main()
        