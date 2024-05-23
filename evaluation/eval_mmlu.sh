source eval.sh

# main evaluation function
eval_mmlu() {
    mdir=$1
    set_save_dir $mdir mmlu
    mkdir -p $save_dir
    cmd="python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $DATA_DIR/mmlu \
    --save_dir $save_dir \
    --model_name_or_path $mdir \
    --tokenizer_name_or_path $mdir \
    --eval_batch_size 4 \
    --convert_to_bf16"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# evaluate the validation set, which is not supported yet
valid_mmlu() {
    mdir=$1
    type=$2
    set_valid_dir $mdir mmlu
    mkdir -p $save_dir
    cmd="python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --eval_valid \
    --data_dir $DATA_DIR/mmlu \
    --save_dir $save_dir \
    --model_name_or_path $mdir \
    --tokenizer_name_or_path $mdir \
    --eval_batch_size 4 \
    --convert_to_bf16"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# extract the results
extract_mmlu() {
    mdir=$1
    set_save_dir $mdir mmlu
    result=$(jq .average_acc $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

# extract the results for the validation set
extract_valid_mmlu() {
    mdir=$1
    set_valid_dir $mdir mmlu
    result=$(jq .average_acc $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

export -f eval_mmlu
export -f valid_mmlu
export -f extract_mmlu
export -f extract_valid_mmlu
