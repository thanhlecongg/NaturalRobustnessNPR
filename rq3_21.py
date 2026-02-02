import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import math 
from utils import *

def evaluate(tool):
    original_results = parse_origin_results(open(f"data/meta/original.json", 'r'), f"data/plausible_patches/original-{tool}.xlsx", tool)
    all_transform_results = {}
    all_plausible, all_correct, all_bugs = {}, {}, {}
    all_positive_changes, all_negative_changes = {}, {}
    all_changes = []
    for transform in ["naming", "expression", "statement"]:
        transform_results = parse_results(transform, open(f"data/meta/{transform}.json", 'r'), f"data/plausible_patches/{transform}-{tool}.xlsx", tool, False)
        all_transform_results.update(transform_results)
        plausible, correct, bugs = normalized_evaluation_metrics(transform_results)
        positive_changes, negative_changes, _, transform_changes = calculate_changes(transform, transform_results, original_results)
        all_plausible, all_correct, all_bugs = merge_dict(all_plausible, plausible), merge_dict(all_correct, correct), merge_dict(all_bugs, bugs)
        all_positive_changes, all_negative_changes = merge_dict(all_positive_changes, positive_changes), merge_dict(all_negative_changes, negative_changes)
        for change in transform_changes:
            all_changes.append(f'{transform}_{change["id"]}')
    return all_changes

def load_naturalness_evaluation(type):
    path = "data/ids/{}_ids.txt".format(type)
    return open(path).read().split("\n")

print("----------------------------------------------------------------------")
print("==> [RQ3.2 - Section 5.2.2] Prediction Changes")
changes = {
    "alpharepair": evaluate("alpharepair"),
    "recoder": evaluate("recoder"),
    "sequencer": evaluate("sequencer"),
    "selfapr": evaluate("selfapr"),
    "rewardrepair": evaluate("rewardrepair"),
    "repairllama": evaluate("repairllama"),
    "incoder": evaluate("incoder")
}

naturalness_evaluation = {
    "unnatural": load_naturalness_evaluation("unnatural"),
    "natural": load_naturalness_evaluation("natural"),
    "likely_natural": load_naturalness_evaluation("likely_natural"),
    "likely_unnatural": load_naturalness_evaluation("likely_unnatural")
}

change_categories = {
    "unnatural": set(),
    "likely_unnatural": set(),
    "likely_natural": set(),
    "natural": set(),
}

naturalness_ratio ={
    "sequencer": [],
    "recoder": [],
    "alpharepair": [],
    "selfapr": [],
    "rewardrepair": [],
    "repairllama": [],
    "incoder": []
}
for tool in changes:  

    for cat in change_categories:
        total = 0
        for change in changes[tool]:
            if change in naturalness_evaluation[cat]:
                change_categories[cat].add(change)
                total += 1
        naturalness_ratio[tool].append(total/len(changes[tool]))
   
for tool in naturalness_ratio:
    assert math.isclose(sum(naturalness_ratio[tool]), 1.0, abs_tol=0.000000000001)

values = np.array(list(naturalness_ratio.values()))
total = values.sum(axis=0)/7
values = np.concatenate((values, total.reshape(1, 4)), axis=0)
values = np.round(values * 100, 1)
categories = ['SequenceR', 'Recorder', 'AlphaRepair', 'SelfAPR', 'RewardRepair', 'RepairLLama', 'Incoder', 'Overall']

print("Drawing Figure 4 ...")
fig, ax = plt.subplots(figsize=(12, 6))
column_width = 0.8

bars1 = ax.bar(categories, values[:, 0], label='Unnatural', edgecolor="black", color="#FF9999", width=column_width)
ax.bar_label(bars1, label_type='center')

bars2 = ax.bar(categories, values[:, 1] + values[:, 2], bottom=values[:, 0], label='Likely Natural/Unnatural', edgecolor="black", color="#4CAF50", width=column_width)
ax.bar_label(bars2, label_type='center')
bars3 = ax.bar(categories, values[:, 3], bottom=values[:, 0] + values[:, 1] + values[:, 2], label='Natural', edgecolor="black", color="#2196F3", width=column_width)
ax.bar_label(bars3, label_type='center')
ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.8)  # Increase bottom margin

# Add labels and a legend
# plt.xlabel('Transformation Level', fontsize=14)
plt.ylabel('Proportion (%)', fontsize=14)
plt.xticks(fontsize=14, rotation=45, ha='right')
plt.yticks(fontsize=14)  # Rotate y-tick labels by 45 degrees

plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
        ncol=3, fancybox=True, shadow=True, fontsize="14")
print("=> Please see results in data/figures/fig4_naturalness.png")
plt.savefig("data/figures/fig4_naturalness.png", dpi=300)

