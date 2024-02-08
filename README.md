# LESS: Selecting Influential Data for Targeted Instruction Tuning
🌟 [ArXiv Preprint](https://arxiv.org/abs/2402.04333) 

This repo hosts the code for the paper "LESS: Selecting Influential Data for Targeted Instruction Tuning". In this work, we propose a data selection method to select influential data to induce a target capability.

## 🔗 Quick Links
- [LESS: Selecting Influential Data for Targeted Instruction Tuning](#less-selecting-influential-data-for-targeted-instruction-tuning)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Preparation](#data-preparation)
  - [Data Selection Pipeline](#data-selection-pipeline)
    - [Step 1: Warmup training](#step-1-warmup-training)
    - [Step 2: Building the gradient datastore](#step-2-building-the-gradient-datastore)
    - [Step 3: Selecting data for a task](#step-3-selecting-data-for-a-task)
    - [Step 4: Train with your selected data](#step-4-train-with-your-selected-data)
  - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)


## Install Requirements
**Step 1**: To get started with this repository, you'll need to follow these installation steps. Before proceeding, make sure you have [Pytorch](https://pytorch.org/get-started/previous-versions/) installed. 
```
pip3 install torch==2.1.2 torchvision torchaudio
```

**Step 2**: Then install the rest of the required packages:
```
cd LESS
pip install -r requirement.txt
```

**Step 3**: Finally, install the `less` package in editable mode to make it accessible for your development environment:
```
pip install -e .
```


## Data Preparation
We follow the [open-instruct](https://github.com/allenai/open-instruct?tab=readme-ov-file#dataset-preparation) repo to prepare hour instruction tuning datasets. In our project, we utilize a combination of four training datasets: Flan v2, COT, Dolly, and Open Assistant. For the purposes of evaluation, we employ three additional datasets: MMLU, Tydiqa, and BBH. A processed version of these files are available [here](https://huggingface.co/datasets/princeton-nlp/less_data).

## Data Selection Pipeline

### Step 1: Warmup training
To enhance downstream performance from data selection, it's crucial to start with a warmup training step. This involves selecting a small portion of your entire dataset to train using the LoRA method. Follow these steps for effective warmup training:

```bash 
DATA_DIR=../data
MODEL_PATH=meta-llama/Llama-2-7b-hf
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"
```

### Step 2: Building the gradient datastore
Once the initial warmup training stage is completed, we will collect gradients for the entire training dataset. For each checkpoint, our goal is to obtain the gradients of all the training data that we would like to select from. An example script is shown below.

```bash
CKPT=105

TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=../data/train/processed/dolly/dolly_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

./less/scripts/get_info/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
```
Ideally, you would aim to create a datastore that encompasses a gradient of all the checkpoints and training data from which you wish to choose. 

### Step 3: Selecting data for a task
To select data for a particular downstream task, it's necessary to first prepare data specific to that task, using the same instruction-tuning prompt format as was employed during training. We have set up data loading modules for three evaluation datasets featured in our work: BBH, TydiQA, and MMLU. If you're interested in data selection for additional tasks, you can expand the [`less/data_selection/get_validation_dataset.py`](less/data_selection/get_validation_dataset.py) script to accommodate those tasks. Similar to obtaining gradients for training data, run the following script. The primary difference is that this process will yield SGD gradients for the validation data, following the formulation of the influence estimation. 

```bash

CKPT=105
TASK=tydiqa
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
DATA_DIR=../data
DIMS="4096 8192" # We use 8192 as our default projection dimension 

./less/scripts/get_info/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
```
You should gain the gradients of the validation data for all the checkpoints you used for building the gradient datastore in the previous step. After obtaining the gradients for the validation data, we can then select data for the task. The following script will calculate the influence score for each training data point, and select the top-k data points with the highest influence score.

```bash
DIM=8192 # decide which dimension to use
GRADIENT_PATH=../grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
CKPTS="105 211 317 420" # checkpoing index
CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch

VALIDATION_GRADIENT_PATH=../grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="tydiqa"
SELECTED_DATA_OUTPUT_PATH="../selected_data"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
```

The influence score for each training data point will be saved in the `OUTPUT_PATH` directory. You can use the following script to select the top-k data points with the highest influence score. 

```bash
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ../data/train/processed/dolly/dolly_data.jsonl ../data/train/processed/oasst1/oasst1_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05
```

### Step 4: Train with your selected data
After selecting the data, you can use the following script to train the model with the selected data. 

```bash 
TARGET_TASK_NAME="tydiqa"
PERCENTAGE=0.05
TRAIN_FILES=../selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
```
Note that you can also perform full-parameter finetuning by removing the lora training parameters. 

## Evaluation
Please follow the instructions in the [evaluation](evaluation/README.md) folder to evaluate the performance of the model trained on the selected data.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Mengzhou (mengzhou@princeton.edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{xia2024less,
  title={Less: Selecting Influential Data for Instruction Tuning},
  author={Xia, Mengzhou and Malladi, Sadhika and Gururangan, Suchin and Arora, Sanjeev and Chen, Danqi},
  year={2024}
}
```




