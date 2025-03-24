# chmod 777 scripts/train_Houston.sh

# Training with different amounts of training data
python train.py --lr 1e-4 --data_set Houston2013 --model Early --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model Middle --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model Late --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model Decision --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model Cross --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model S2E --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model End --device cuda:0 --data_dir ./data
python train.py --lr 1e-4 --data_set Houston2013 --model FusAt --device cuda:0 --data_dir ./data
python train.py --lr 2e-4 --data_set Houston2013 --model HGN --device cuda:0 --data_dir ./data