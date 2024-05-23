source eval.sh

# main evaluation function
eval_truthfulqa() {
    mdir=$1
    set_save_dir $mdir truthfulqa
    mkdir -p $save_dir
    cmd="python -m eval.truthfulqa.run_eval \
    --data_dir $DATA_DIR/truthfulqa/ \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --use_chat_format \
    --eval_batch_size 20 \
    --preset qa \
    --metrics mc \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# extract the results
extract_truthfulqa() {
    mdir=$1
    set_save_dir $mdir truthfulqa
    echo $save_dir
    mc1=$(jq .MC1 $save_dir/metrics.json)
    mc1=$(echo "$mc1 * 100" | bc)
    mc2=$(jq .MC2 $save_dir/metrics.json)
    mc2=$(echo "$mc2 * 100" | bc)
    echo $mc2
}

export -f eval_truthfulqa
export -f extract_truthfulqa

