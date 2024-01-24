#!/bin/bash
#SBATCH --job-name=eval_loss
#SBATCH --output=/scratch/gpfs/mengzhou/space10/output/logs/eval_loss-%j.out
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
    output_path=${model_dir}/eval_loss/${task}/step${step} # path to output
    model=${model_dir}/checkpoint-${step} # path to model
    train_batch=${model_dir}/train_batch_${step}.pt # train batch size
    
    if [[ ! -d $output_path ]]; then
        mkdir -p $output_path
    fi
    
    # get loss
    python3 -m run.first_order_checking.calculate_loss \
    --task $task \
    --info_type loss \
    --model_path $model \
    --output_path $output_path \
    --data_dir $data_dir \
    --train_batch $train_batch
done

"""
for task in bbh tydiqa mmlu; do
sbatch -p cli run/first_order_checking/calculate_loss.sh $task
done
"""
