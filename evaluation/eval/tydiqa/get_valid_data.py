import json
import os
import random

random.seed(42)

data_dir = "/scratch/gpfs/mengzhou/space10/data/eval/tydiqa"
data = json.load(open(
    "/scratch/gpfs/mengzhou/space10/data/eval/tydiqa/tydiqa-goldp-v1.1-train.json", "r"))

test_data = []
with open(os.path.join(data_dir, "tydiqa-goldp-v1.1-dev.json")) as fin:
    dev_data = json.load(fin)
    for article in dev_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                example = {
                    "id": qa["id"],
                    "lang": qa["id"].split("-")[0],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": qa["answers"]
                }
                test_data.append(example)

data_languages = set([example["lang"] for example in test_data])
train_data_for_langs = {lang: [] for lang in data_languages}
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            lang = qa["id"].split("-")[0]
            if lang in data_languages:
                example = {
                    "id": qa["id"],
                    "lang": lang,
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": qa["answers"]
                }
                train_data_for_langs[lang].append(example)
for lang in data_languages:
    # sample n_shot examples from each language
    train_data_for_langs[lang] = random.sample(
        train_data_for_langs[lang], 1)

ids = []
for lang in data_languages:
    for example in train_data_for_langs[lang]:
        ids.append(example["id"])

data_with_ids = []
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if qa["id"] in ids:
                data_with_ids.append(article)

with open("/scratch/gpfs/mengzhou/space10/data/eval/tydiqa/one-shot-valid/tydiqa-goldp-v1.1-dev.json", "w", encoding="utf-8") as f:
    f.write(json.dumps({"data": data_with_ids}, ensure_ascii=False, indent=4))

test_data = {}
for article in data_with_ids:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            lang = qa["id"].split("-")[0]
            example = {
                "id": qa["id"],
                "lang": lang,
                "context": paragraph["context"],
                "question": qa["question"],
                "answers": qa["answers"]
            }
            test_data[lang] = [example]

with open("/scratch/gpfs/mengzhou/space10/data/eval/tydiqa/one-shot-valid/tydiqa-goldp-v1.1-dev-examples.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(test_data, ensure_ascii=False, indent=4))
