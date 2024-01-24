from transformers import AutoTokenizer

from less.data_selection.get_validation_dataset import (get_dataloader,
                                                        get_dataset)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for task in ["bbh", "tydiqa", "mmlu"]:
    dataset = get_dataset(task,
                      data_dir="../data",
                      tokenizer=tokenizer,
                      max_length=2048)
    dataset.save_to_disk(f"/scratch/gpfs/mengzhou/space10/data/few_shot_mistral/{task}.pt")