source eval.sh

# main evaluation function
eval_tydiqa() {
    mdir=$1
    set_save_dir $mdir tydiqa
    mkdir -p $save_dir
    cmd="python -m eval.tydiqa.run_eval \
    --data_dir $DATA_DIR/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 200 \
    --max_context_length 512 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 20 \
    --use_chat_format \
    --convert_to_bf16 \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# evaluate the validation set, which is not supported yet
valid_tydiqa() {
    mdir=$1
    set_valid_dir $mdir tydiqa
    mkdir -p $save_dir
    cmd="python -m eval.tydiqa.run_eval \
    --data_dir $DATA_DIR/tydiqa/one-shot-valid \
    --n_shot 0 \
    --eval_valid \
    --max_num_examples_per_lang 200 \
    --max_context_length 512 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 20 \
    --use_chat_format \
    --convert_to_bf16 \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# extract the results
extract_tydiqa() {
    mdir=$1
    set_save_dir $mdir tydiqa
    result=$(jq .average.f1 $save_dir/metrics.json)
    echo $result
}

# extract the results for the validation set
extract_valid_tydiqa() {
    mdir=$1
    set_valid_dir $mdir tydiqa
    result=$(jq .average.f1 $save_dir/metrics.json)
    echo $result
}

export -f eval_tydiqa
export -f valid_tydiqa
export -f extract_tydiqa
export -f extract_valid_tydiqa
