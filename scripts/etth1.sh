model_name=TimeCNN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --dropout 0.6\
  --conv_dropout 0.8\
  --learning_rate 0.0005\
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --dropout 0.6\
  --conv_dropout 0.8\
  --learning_rate 0.0005\
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2\
  --enc_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --dropout 0.6\
  --conv_dropout 0.8\
  --learning_rate 0.0005\
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2\
  --enc_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --dropout 0.6\
  --conv_dropout 0.8\
  --learning_rate 0.0005\
  --itr 1