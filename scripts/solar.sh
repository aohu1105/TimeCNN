
model_name=TimeCNN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --dropout 0.3\
  --conv_dropout 0.3\
  --use_norm 0\
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --dropout 0.3\
  --conv_dropout 0.3\
  --use_norm 0\
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --dropout 0.3\
  --conv_dropout 0.3\
  --use_norm 0\
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --dropout 0.3\
  --conv_dropout 0.3\
  --use_norm 0\
  --learning_rate 0.0005 \
  --itr 1

