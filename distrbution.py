from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="ticks", color_codes=True)

dataset = load_dataset('dane')["train"]

all_tags = [
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
counts = [0] * len(all_tags)

for row in dataset:
    ner_tags = row["ner_tags"]

    for ner in ner_tags:
        if ner == 0:
            continue
        counts[ner-1] += 1

count_dict = dict([(all_tags[t], c) for t, c in enumerate(counts)])

ind = np.arange(len(count_dict))
plt.bar(ind, list(count_dict.values()))
plt.xticks(ind, list(count_dict.keys()))
plt.ylabel("Count")
plt.xlabel("NER_TAGS")
plt.tight_layout()
plt.savefig("results/distr.png")
plt.show()