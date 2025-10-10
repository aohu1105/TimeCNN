if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/pems03" ]; then
    mkdir ./logs/pems03
fi
export CUDA_VISIBLE_DEVICES=0

seq_len=96
model_name=FilterTS

pred_len=96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --factor 3 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 256 \
  --quantile 0.9 \
  --bandwidth 1 \
  --top_K_static_freqs 10 \
  --filter_type all \
  --learning_rate 0.005 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --lradj type1 \
  --use_norm 0 \
  --itr 1 >logs/pems03/$model_name'_'pems03_$seq_len'_'$pred_len.log

pred_len=192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --factor 3 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --quantile 0.9 \
  --bandwidth 1 \
  --top_K_static_freqs 10 \
  --filter_type all \
  --learning_rate 0.001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --lradj type1 \
  --use_norm 0 \
  --itr 1 >logs/pems03/$model_name'_'pems03_$seq_len'_'$pred_len.log

pred_len=336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --factor 3 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --quantile 0.9 \
  --bandwidth 1 \
  --top_K_static_freqs 10 \
  --filter_type all \
  --learning_rate 0.001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --lradj type1 \
  --use_norm 0 \
  --itr 1 >logs/pems03/$model_name'_'pems03_$seq_len'_'$pred_len.log

pred_len=720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --factor 3 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --quantile 0.9 \
  --bandwidth 1 \
  --top_K_static_freqs 10 \
  --filter_type all \
  --learning_rate 0.001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --lradj type1 \
  --use_norm 0 \
  --itr 1 >logs/pems03/$model_name'_'pems03_$seq_len'_'$pred_len.log