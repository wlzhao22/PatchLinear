export export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Anomaly_Detection" ]; then
    mkdir ./logs/Anomaly_Detection
fi
seq_len=144
model_name=DLinear

root_path_name=./dataset/KPI/
data_path_name=KPI
model_id_name=KPI

random_seed=2023

python -u run_longExp.py \
    --random_seed $random_seed \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len$name \
    --model $model_name \
    --data KPI \
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
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 10\
    --dp_window 1440\
    --score_calc mae\
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/Anomaly_Detection/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len$name.log 
