#!/bin/bash
#SBATCH --job-name=train_grad_fixadam
#SBATCH --output=/scratch/gpfs/mengzhou/space10/output/logs/train_grad_fixadam-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --constraint gpu80

cd /scratch/gpfs/mengzhou/space10/final/DIGSIT


type=adam

for step in {33..105}; do
    data_dir=../data
    model_dir=$n/space10/out/46_train_for_analysis/p0.05_seed3_lora/
    output_path=${model_dir}/train_${type}_grad_fixadam/step${step} # path to output
    model=${model_dir}/checkpoint-${step} # path to model
    train_batch=${model_dir}/data_batch_${step}.pt # train batch size
    optimizer_state=$n/space10/out/46_train_for_analysis/p0.05_seed6_lora/checkpoint-105
    
    # get adam grads
    dims="8192"
    gradient_type=$type
    train_file="../data/train/processed/flan_v2/flan_v2_data.jsonl ../data/train/processed/cot/cot_data.jsonl  ../data/train/processed/dolly/dolly_data.jsonl ../data/train/processed/oasst1/oasst1_data.jsonl"
    python3 -m run.first_order_checking.calculate_loss \
    --train_file $train_file \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type $gradient_type \
    --train_batch $train_batch \
    --optimizer_state_path $optimizer_state
done

"""
sbatch -p cli run/first_order_checking/calculate_train_grad_fixadam.sh
"""
