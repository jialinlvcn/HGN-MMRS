# chmod 777 scripts/train_Augsburg.sh

# Training with different amounts of training data
python train.py --lr 1e-4 --data_set Augsburg --model Early --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Augsburg --model Middle --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Augsburg --model Late --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Augsburg --model Decision --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Augsburg --model S2E --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Augsburg --model End --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Augsburg --model FusAt --device cuda:0 --data_dir ./data
python train.py --lr 2e-4 --data_set Augsburg --model HGN --device cuda:0 --data_dir ./data
