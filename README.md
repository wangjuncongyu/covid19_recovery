# covid19_recovery
tensorflow projects for recovery-time prediction of COVID-19

> This is a deep-learning framework for identificating high-risk COVID-19 patients and estimating how long the patient can be cured。

### requirments
- Anaconda python 3.7.3 Win10
- Tensorflow 2.0.0 with GPU

## network architecture (see achitecture.pptx)


#if you have any problem, please feel free to ask questions via sending email to wjcy19870122@sjtu.edu.cn
## Training

``` bash
(1) prepare your data (see the 50 data subjects for examples).
(2) cd trainers and run the file: run_train.bat.
Note: you may need to modify the configs/cfgs.py file:changing cfg.data_set to the directory of your dataset.
```
##  Evaluation
``` bash
(1) cd tests and run the run_test.bat file.
(2) run the compute_metrics.py file to obtain the results.
```

