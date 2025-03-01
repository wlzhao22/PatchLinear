export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Anomaly_Detection" ]; then
    mkdir ./logs/Anomaly_Detection
fi
seq_len=24
model_name=iTransformer

root_path_name=./dataset/Yahoo/
data_path_name=Yahoo
model_id_name=Yahoo

random_seed=2023

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --task_name anomaly_detection \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data Yahoo \
    --features S \
    --seq_len $seq_len \
    --pred_len 1 \
    --enc_in 1 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 128 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --target value\
    --patch_len 12\
    --stride 4\
    --des 'Exp' \
    --train_epochs 30\
    --dp_window 48\
    --dis_ratio 48\
    --cache_window 240\
    --score_calc mae\
    --delay 3\
    --ds_thresh 0.7\
    --plot 'n'\
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/Anomaly_Detection/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
  

