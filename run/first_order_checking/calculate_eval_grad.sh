#!/bin/bash
#SBATCH --job-name=eval_grad
#SBATCH --output=/scratch/gpfs/mengzhou/space10/output/logs/eval_grad-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --constraint gpu80

cd /scratch/gpfs/mengzhou/space10/final/DIGSIT

task=$1

for step in {1..105}; do
    data_dir=../data
    model_dir=$n/space10/out/46_train_for_analysis/p0.05_seed3_lora/
    output_path=${model_dir}/eval_sgd_grad/${task}/step${step} # path to output
    model=${model_dir}/checkpoint-${step} # path to model

    mkdir $output_path 
    dims=8192
    # get sgd grads
    python3 -m run.first_order_checking.calculate_loss \
    --data_dir $data_dir \
    --task $task \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type sgd
done

"""
for task in bbh tydiqa mmlu; do
sbatch -p cli run/first_order_checking/calculate_eval_grad.sh $task
done
"""