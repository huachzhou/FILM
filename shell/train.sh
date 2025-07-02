echo $1
output_dir="./output"
base_model="../llama-7b-hf/"
train_data="./data/sample_data/train.json"
val_data="./data/sample_data/test.json"
instruction_model="./alpaca-lora-7B/alpaca-lora-7B/checkpoint-600/"
for lr in 1e-4
do
    for dropout in 0.2
    do
        for sample in 64
        do
                mkdir -p $output_dir
                echo "output_dir: $output_dir, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1  python -u train.py \
                    --base_model $base_model \
                    --model_type 'FILM' \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir ${output_dir} \
                    --train_batch_size 32 \
                    --test_batch_size 52 \
                    --num_epochs 40 \
                    --learning_rate $lr \
                    --cutoff_len 1024 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs False\
                    --group_by_length True\
                    --resume_from_checkpoint $instruction_model
        done
    done
done
