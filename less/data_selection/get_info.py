"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse
import os
import pdb
from copy import deepcopy
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from less.data_selection.collect_grad_reps import (collect_grads, collect_reps,
                                                   get_loss)
from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import (get_dataloader,
                                                        get_dataset)


def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument('--task', type=str, default=None,
                    help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
parser.add_argument("--train_file", type=str,
                    default=None, help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
parser.add_argument(
    "--info_type", choices=["grads", "reps", "loss"], help="The type of information")
parser.add_argument("--model_path", type=str,
                    default=None, help="The path to the model")
parser.add_argument("--max_samples", type=int,
                    default=None, help="The maximum number of samples")
parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16"], help="The torch data type")
parser.add_argument("--output_path", type=str,
                    default=None, help="The path to the output")
parser.add_argument("--data_dir", type=str,
                    default=None, help="The path to the data")
parser.add_argument("--gradient_projection_dimension", nargs='+',
                    help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--gradient_type", type=str, default="adam",
                    choices=["adam", "sign", "sgd"], help="The type of gradient")
parser.add_argument("--chat_format", type=str,
                    default="tulu", help="The chat format")
parser.add_argument("--use_chat_format", type=bool,
                    default=True, help="Whether to use chat format")
parser.add_argument("--max_length", type=int, default=2048,
                    help="The maximum length")
parser.add_argument("--zh", default=False, action="store_true",
                    help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
parser.add_argument("--initialize_lora", default=False, action="store_true",
                    help="Whether to initialize the base model with lora, only works when is_peft is False")
parser.add_argument("--lora_r", type=int, default=8,
                    help="The value of lora_r hyperparameter")
parser.add_argument("--lora_alpha", type=float, default=32,
                    help="The value of lora_alpha hyperparameter")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="The value of lora_dropout hyperparameter")
parser.add_argument("--lora_target_modules", nargs='+', default=[
                    "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")

args = parser.parse_args()
assert args.task is not None or args.train_file is not None

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
model = load_model(args.model_path, dtype)

# pad token is not added by default for pretrained models
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# resize embeddings if needed (e.g. for LlamaTokenizer)
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

if args.initialize_lora:
    assert not isinstance(model, PeftModel)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

if isinstance(model, PeftModel):
    model.print_trainable_parameters()

adam_optimizer_state = None
if args.info_type == "grads" and args.gradient_type == "adam":
    optimizer_path = os.path.join(args.model_path, "optimizer.bin")
    adam_optimizer_state = torch.load(
        optimizer_path, map_location="cpu")["state"]

if args.task is not None:
    dataset = get_dataset(args.task,
                          data_dir=args.data_dir,
                          tokenizer=tokenizer,
                          chat_format=args.chat_format,
                          use_chat_format=args.use_chat_format,
                          max_length=args.max_length,
                          zh=args.zh)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
else:
    assert args.train_file is not None
    dataset = get_training_dataset(
        args.train_file, tokenizer, args.max_length, sample_percentage=1.0)
    columns = deepcopy(dataset.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    dataset = dataset.remove_columns(columns)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)

if args.info_type == "reps":
    collect_reps(dataloader, model, args.output_path,
                 max_samples=args.max_samples)
elif args.info_type == "grads":
    collect_grads(dataloader,
                  model,
                  args.output_path,
                  proj_dim=args.gradient_projection_dimension,
                  gradient_type=args.gradient_type,
                  adam_optimizer_state=adam_optimizer_state,
                  max_samples=args.max_samples)
elif args.info_type == "loss":
    get_loss(dataloader, model, args.output_path)
