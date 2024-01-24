cd $n/space10/final/DIGSIT

for task in mmlu bbh tydiqa; do
    data_dir=$n/space10/final/data
    # model=mistralai/Mistral-7B-v0.1
    # model_name=mistral-7b
    model=meta-llama/Llama-2-13b-hf
    model_name=llama-2-13b-hf
    output_path=$n/space10/final/loss/${model_name}/$task/${model_name}
    if [[ ! -d $output_path ]]; then
        mkdir -p $output_path
    fi
    loss_file=$output_path/loss.txt
    if [[ ! -f $loss_file ]]; then
        bash ./less/scripts/get_info/loss/get_eval_pretrain_loss.sh $task $data_dir $model $output_path
    fi
done

# lora random selection
base_model_name=llama-2-13b-hf
for task in mmlu bbh tydiqa; do
    for seed in 3 6 9; do
        for ckpt in 105 211 317 420; do
            data_dir=$n/space10/final/data
            model=$n/space10/out/11_13b_train/llama2-13b_lora_p0.05_seed${seed}/checkpoint-${ckpt}
            model_name=p0.05_seed${seed}_lora_ckpt${ckpt}_random
            output_path=$n/space10/final/loss/${base_model_name}/$task/${model_name}
            if [[ ! -d $output_path ]]; then
                mkdir -p $output_path
            fi
            loss_file=$output_path/loss.txt
            if [[ ! -f $loss_file ]]; then
                bash ./less/scripts/get_info/loss/get_eval_lora_loss.sh $task $data_dir $model $output_path
            fi
        done
    done
done

# lora selected data with less
base_model_name=llama-2-13b-hf
for task in mmlu bbh tydiqa; do
    for seed in 3 6 9; do
        for ckpt in 105 211 317 420; do
            data_dir=$n/space10/final/data
            if [[ $task == "mmlu" ]]; then task_name=mmlu-chat;
                elif [[ $task == "bbh" ]]; then task_name=bbh-icl;
        else task_name=$task; fi
            model=$n/space10/out/18_13b_select_seed/${task_name}_13b_adam_sim_trainp0.05_seed${seed}_p0.05_seed0_lora/checkpoint-${ckpt}
            model_name=p0.05_seed${seed}_lora_ckpt${ckpt}_less
            output_path=$n/space10/final/loss/${base_model_name}/$task/${model_name}
            if [[ ! -d $output_path ]]; then
                mkdir -p $output_path
            fi
            loss_file=$output_path/loss.txt
            if [[ ! -f $loss_file ]]; then
                bash ./less/scripts/get_info/loss/get_eval_lora_loss.sh $task $data_dir $model $output_path
            fi
        done
    done
done