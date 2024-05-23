source eval.sh

# main evaluation function
eval_gsm8k() {
    mdir=$1
    set_save_dir $mdir gsm8k
    echo $save_dir
    mkdir -p $save_dir
    cmd="python -m eval.gsm.run_eval \
    --data_dir $DATA_DIR/gsm/ \
    --n_shot 8 \
    --max_num_examples 200 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# extract the results
extract_gsm8k() {
    mdir=$1
    set_save_dir $mdir gsm8k
    result=$(jq .exact_match $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

export -f eval_gsm8k
export -f extract_gsm8k

