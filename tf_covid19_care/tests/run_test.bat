python test_network.py -train_subsets=1;2;3;4 -eval_subsets=subset_5.csv -batch_size=72  -save_root=huoshenshan_rst_org -save_flag=5 -treatment=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -gt_ctimages=1
python test_network.py -train_subsets=1;2;3;5 -eval_subsets=subset_4.csv -batch_size=72  -save_root=huoshenshan_rst_org -save_flag=4 -treatment=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -gt_ctimages=1
python test_network.py -train_subsets=1;2;4;5 -eval_subsets=subset_3.csv -batch_size=72  -save_root=huoshenshan_rst_org -save_flag=3 -treatment=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -gt_ctimages=1
python test_network.py -train_subsets=1;3;4;5 -eval_subsets=subset_2.csv -batch_size=72  -save_root=huoshenshan_rst_org -save_flag=2 -treatment=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -gt_ctimages=1
python test_network.py -train_subsets=2;3;4;5 -eval_subsets=subset_1.csv -batch_size=72  -save_root=huoshenshan_rst_org -save_flag=1 -treatment=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -gt_ctimages=1


