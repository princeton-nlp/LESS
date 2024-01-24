
seed=3
cd $n/space10/final/DIGSIT

DIM=8192 # decide which dimension to use
GRADIENT_PATH=/scratch/gpfs/mengzhou/space10/grads/7b_trainp_adam_grads/p0.05_seed${seed}/adam_grads_llama2-7b-p0.05_seed${seed}_{}_{}_dim8192/all_unormaize.pt
TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
CKPTS="105 211 317 420" # checkpoing index
CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch

VALIDATION_GRADIENT_PATH=/scratch/gpfs/mengzhou/space10/grads/7b_trainp_adam_grads/p0.05_seed${seed}/few_shot_grads/grads_llama2-7b-p0.05_seed3_{}_bbh-icl_dim8192/all_unormaize.pt
TARGET_TASK_NAMES="bbh"
SELECTED_DATA_OUTPUT_PATH="../selected_data/unnormalized_gradients"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

TARGET_TASK_NAMES="bbh"
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ../data/train/processed/flan_v2/flan_v2_data.jsonl ../data/train/processed/cot/cot_data.jsonl  ../data/train/processed/dolly/dolly_data.jsonl ../data/train/processed/oasst1/oasst1_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05


# train
export WANDB_MODE="offline"
task=tydiqa
TRAIN_FILES=../selected_data/unnormalized_gradients/${task}/top_p0.05.jsonl
model_path=meta-llama/Llama-2-7b-hf
job_name=llama2-7b-p0.05_seed${seed}_${task}_unnormalized
output_dir=../out/${job_name}
sbatch -p cli --gres=gpu:4 --output ../out/slurm/%j-%x.out --job-name $job_name --mem=200g -t 2:00:00 ./less/scripts/train/train.sh "$TRAIN_FILES" "$model_path" "$job_name"


# task=tydiqa
# PROJ_DIR=$n/space10
# train_file_dir=${PROJ_DIR}/data/split_train_dev_llama2
# job_name=${task}_adam_sim_trainp0.05_seed3

# TRAIN_FILES=${train_file_dir}/$job_name/${job_name}_p0.05.jsonl
# model_path=meta-llama/Llama-2-7b-hf
# job_name=llama2-7b-p0.05_seed${seed}_${task}_normalized
# output_dir=../out/${job_name}
# bash ./less/scripts/train/train.sh "$TRAIN_FILES" "$model_path" "$job_name"