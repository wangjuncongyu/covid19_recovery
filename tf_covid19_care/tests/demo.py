import _init_pathes

from CAD_wrapper.CAD_pipeline import CAD_Pipeline

pipeline = CAD_Pipeline()
pipeline.initialize()
pipeline.predict_risk_for(r'D:\data\data_root\patient_0000232')