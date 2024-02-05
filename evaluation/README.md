## Evaluation

We mainly employ three evaluation datasets to assess the performance of our data selection pipeline: **MMLU**, **Tydiqa**, and **BBH**. We use the evaluation pipeline [open-instruct](https://github.com/allenai/open-instruct/tree/main/eval). To evaluate a trained model, please follow the steps below:

### Step 1: Install Open-Instruct
```bash
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
pip install -e .
```

### Step 2: Evaluation
Please check out the `eval_mmlu.sh`, `eval_tydiqa.sh`, and `eval_bbh.sh` scripts in the `evaluation` directory. These scripts contain the necessary commands to evaluate the model on the respective datasets. 

