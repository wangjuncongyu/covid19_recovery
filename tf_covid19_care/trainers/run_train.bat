python train_network.py -train_subsets=1;2;3;4 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=15 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1
python train_network.py -train_subsets=1;2;3;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=15 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1
python train_network.py -train_subsets=1;2;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=15 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1
python train_network.py -train_subsets=1;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=15 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1
python train_network.py -train_subsets=2;3;4;5 -eval_subsets=eval_sets.csv -batch_size=72 -lr=0.001 -nepoches=15 -load_pretrain=0 -gt_treatment=1 -gt_ctimages=1




