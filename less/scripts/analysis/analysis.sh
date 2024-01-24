#!/bin/bash

gradient_path=$1
training_file_names=$2
ckpts=$3
checkpoint_weights=$4

validation_gradient_path=$5
select_task_names=$6
output_path=$7

python3 -m less.data_selection.matching \
--gradient_path $gradient_path \
--training_file_names $training_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--select_task_names $select_task_names \
--output_path $output_path
