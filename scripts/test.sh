# chmod 777 scripts/test.sh

# Test Performance
python test.py --data_set Augsburg --model Early --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model Middle --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model Late --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model Decision --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model S2E --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model End --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model FusAt --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Augsburg --model HGN --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints

# Test Performance
python test.py --data_set Houston2013 --model Early --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model Middle --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model Late --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model Decision --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model Cross --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model S2E --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model End --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model FusAt --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
python test.py --data_set Houston2013 --model HGN --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints