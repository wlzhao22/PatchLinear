export CUDA_VISIBLE_DEVICES=1
if [ ! -d './logs' ]; then
    mkdir ./logs
fi
if [ ! -d './logs/Anomaly_Detection' ]; then
    mkdir ./logs/Anomaly_Detection
fi
seq_len=144
model_name=PatchTST
dir_name=MSL
root_path_name=./dataset/NASA


model_id_name=MSL
random_seed=2023
if [ ! -d './logs/Anomaly_Detection/MSL/PatchTST' ]; then
    mkdir ./logs/Anomaly_Detection/MSL/PatchTST
fi

    python -u run_longExp.py \
      --random_seed $random_seed \
      --task_name anomaly_detection \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path MSL \
      --model_id $model_id_name_$seq_len \
      --model $model_name \
      --data MSL\
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
      --score_calc mae\
      --dp_window 360\
      --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/Anomaly_Detection/$dir_name/$model_name/$model_name'_'$model_id_name'_'$name'_'$seq_len.log 