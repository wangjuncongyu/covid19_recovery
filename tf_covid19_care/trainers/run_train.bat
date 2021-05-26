python train_network.py -train_subsets=1;2;3;4 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1 -model_path=fold5_treat_im
python train_network.py -train_subsets=1;2;3;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1 -model_path=fold4_treat_im
python train_network.py -train_subsets=1;2;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1 -model_path=fold3_treat_im
python train_network.py -train_subsets=1;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1 -model_path=fold2_treat_im
python train_network.py -train_subsets=2;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1 -model_path=fold1_treat_im

python train_network.py -train_subsets=1;2;3;4 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=0 -gt_ctimages=1 -model_path=fold5_notreat_im
python train_network.py -train_subsets=1;2;3;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=0 -gt_ctimages=1 -model_path=fold4_notreat_im
python train_network.py -train_subsets=1;2;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=0 -gt_ctimages=1 -model_path=fold3_notreat_im
python train_network.py -train_subsets=1;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=0 -gt_ctimages=1 -model_path=fold2_notreat_im
python train_network.py -train_subsets=2;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=0 -gt_ctimages=1 -model_path=fold1_notreat_im

python train_network.py -train_subsets=1;2;3;4 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=0 -model_path=fold5_treat_noim
python train_network.py -train_subsets=1;2;3;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=0 -model_path=fold4_treat_noim
python train_network.py -train_subsets=1;2;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=0 -model_path=fold3_treat_noim
python train_network.py -train_subsets=1;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=0 -model_path=fold2_treat_noim
python train_network.py -train_subsets=2;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=20 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=0 -model_path=fold1_treat_noim




