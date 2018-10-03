python -m basic.cli --mode train --noload --batch_size 400 --sent_size_th 60 --num_steps 0 --num_epochs 3 --len_opt --cluster --num_gpus 1 --run_id $1 --data_dir data/$1 --eval_period 500
