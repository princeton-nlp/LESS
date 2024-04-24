import os
import json
import argparse
import logging
import random
import torch
import datasets
import vllm
from alpaca_eval import evaluate as alpaca_farm_evaluate
from eval.utils import query_openai_chat_model, query_openai_model, generate_completions, dynamic_import_function, load_hf_lm_and_tokenizer
import argparse
import openai

# this is for running alpacaeval! 

# openai.api_key = "7cf72d256d55479383ab6db31cda2fae"
# openai.api_base =  "https://pnlpopenai2.openai.azure.com/" 
# openai.api_type = 'azure'
# openai.api_version = '2023-05-15' # this may change in the future

openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future
openai.api_key = "050fd3ed1d8740bfbd07334dfbc6a614" 
openai.api_base = "https://pnlpopenai3.openai.azure.com/"

# changed model to engine in openai.py in alpaca-eval
def evaluate(args):
    df_leaderboard, annotations = alpaca_farm_evaluate(
            model_outputs=args.output_path,
            reference_outputs=args.reference_path,
            annotators_config="alpaca_eval_gpt4",
            output_path=args.save_dir,
            is_return_instead_of_print=True,
        )

    print(df_leaderboard.to_string(float_format="%.2f"))

    # save to json
    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump(df_leaderboard.to_dict(), fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to the reference outputs. "
                "Alpaca_eval leaderboard use davinci_003 to generate the reference outputs, "
                "but they limit the max_tokens to 300. Here we regenerated reference outputs with max_tokens=2048.",
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    evaluate(args)