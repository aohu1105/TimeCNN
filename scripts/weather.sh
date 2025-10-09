
model_name=TimeCNN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --des 'Exp' \
  --dropout 0.4\
  --d_model 512\
  --d_ff 512\
  --batch_size 16\
  --learning_rate 0.0001\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 21 \
  --des 'Exp' \
  --dropout 0.4\
  --d_model 512\
  --d_ff 512\
  --batch_size 16\
  --learning_rate 0.0001\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 21 \
  --des 'Exp' \
  --dropout 0.4\
  --d_model 512\
  --d_ff 512\
  --batch_size 16\
  --learning_rate 0.0001\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 21 \
  --des 'Exp' \
  --dropout 0.4\
  --d_model 512\
  --d_ff 512\
  --batch_size 16\
  --learning_rate 0.0001\
  --itr 1