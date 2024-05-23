source eval.sh

# main evaluation function
eval_bbh() {
    mdir=$1
    set_save_dir $mdir bbh
    mkdir -p $save_dir
    cmd="python -m eval.bbh.run_eval \
    --data_dir $DATA_DIR/bbh \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 10 \
    --convert_to_bf16 \
    --max_num_examples_per_task 40 " 
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# evaluate the validation set, which is not supported yet
valid_bbh() {
    mdir=$1
    set_valid_dir $mdir bbh
    echo $save_dir
    mkdir -p $save_dir
    cmd="python -m eval.bbh.run_eval \
    --data_dir $DATA_DIR/bbh-valid \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 10 \
    --convert_to_bf16 \
    --eval_valid \
    --max_num_examples_per_task 3 "
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# extract the results
extract_bbh() {
    mdir=$1
    set_save_dir $mdir bbh
    result=$(jq .average_exact_match $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

# extract the results for the validation set
extract_valid_bbh() {
    mdir=$1
    set_valid_dir $mdir bbh
    result=$(jq .average_exact_match $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}


export -f eval_bbh
export -f valid_bbh
export -f extract_bbh
export -f extract_valid_bbh
